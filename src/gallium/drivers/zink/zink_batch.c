#include "zink_batch.h"

#include "zink_context.h"
#include "zink_fence.h"
#include "zink_framebuffer.h"
#include "zink_query.h"
#include "zink_program.h"
#include "zink_render_pass.h"
#include "zink_resource.h"
#include "zink_screen.h"

#include "util/hash_table.h"
#include "util/u_debug.h"
#include "util/set.h"

void
debug_describe_zink_batch_state(char *buf, const struct zink_batch_state *ptr)
{
   sprintf(buf, "zink_batch_state");
}

void
zink_reset_batch_state(struct zink_context *ctx, struct zink_batch_state *bs)
{
   zink_fence_reference(screen, &batch->fence, NULL);

   zink_render_pass_reference(screen, &batch->rp, NULL);
   zink_framebuffer_reference(screen, &batch->fb, NULL);
   set_foreach(batch->programs, entry) {
      struct zink_gfx_program *prog = (struct zink_gfx_program*)entry->key;
      zink_gfx_program_reference(screen, &prog, NULL);
   }
   _mesa_set_clear(batch->programs, NULL);

   /* unref all used resources */
   set_foreach(batch->resources, entry) {
      struct pipe_resource *pres = (struct pipe_resource *)entry->key;
      pipe_resource_reference(&pres, NULL);
   }
   _mesa_set_clear(batch->resources, NULL);

   /* unref all used sampler-views */
   set_foreach(batch->sampler_views, entry) {
      struct pipe_sampler_view *pres = (struct pipe_sampler_view *)entry->key;
      pipe_sampler_view_reference(&pres, NULL);
   }
   _mesa_set_clear(batch->sampler_views, NULL);

   util_dynarray_foreach(&batch->zombie_samplers, VkSampler, samp) {
      vkDestroySampler(screen->dev, *samp, NULL);
   }
   util_dynarray_clear(&bs->zombie_samplers);
   util_dynarray_clear(&bs->persistent_resources);

   set_foreach(bs->desc_sets, entry) {
      struct zink_descriptor_set *zds = (void*)entry->key;
      zink_batch_usage_unset(&zds->batch_uses, bs->fence.batch_id);
      /* reset descriptor pools when no bs is using this program to avoid
       * having some inactive program hogging a billion descriptors
       */
      pipe_reference(&zds->reference, NULL);
      zink_descriptor_set_recycle(zds);
      _mesa_set_remove(bs->desc_sets, entry);
   }

   set_foreach(bs->programs, entry) {
      struct zink_program *pg = (struct zink_program*)entry->key;
      if (pg->is_compute) {
         struct zink_compute_program *comp = (struct zink_compute_program*)pg;
         bool in_use = comp == ctx->curr_compute;
         if (zink_compute_program_reference(screen, &comp, NULL) && in_use)
            ctx->curr_compute = NULL;
      } else {
         struct zink_gfx_program *prog = (struct zink_gfx_program*)pg;
         bool in_use = prog == ctx->curr_program;
         if (zink_gfx_program_reference(screen, &prog, NULL) && in_use)
            ctx->curr_program = NULL;
      }
      _mesa_set_remove(bs->programs, entry);
   }

   set_foreach(bs->fbs, entry) {
      struct zink_framebuffer *fb = (void*)entry->key;
      zink_framebuffer_reference(screen, &fb, NULL);
      _mesa_set_remove(bs->fbs, entry);
   }

   bs->flush_res = NULL;

   bs->descs_used = 0;
   ctx->resource_size -= bs->resource_size;
   bs->resource_size = 0;

   /* only reset submitted here so that tc fence desync can pick up the 'completed' flag
    * before the state is reused
    */
   bs->fence.submitted = false;
   bs->fence.batch_id = 0;
}

static void
reset_batch(struct zink_context *ctx, struct zink_batch *batch)
{
   bs->fence.completed = true;
   zink_reset_batch_state(ctx, bs);
}

void
zink_batch_reset_all(struct zink_context *ctx)
{
   simple_mtx_lock(&ctx->batch_mtx);
   hash_table_foreach(&ctx->batch_states, entry) {
      struct zink_batch_state *bs = entry->data;
      bs->fence.completed = true;
      zink_reset_batch_state(ctx, bs);
      _mesa_hash_table_remove(&ctx->batch_states, entry);
      util_dynarray_append(&ctx->free_batch_states, struct zink_batch_state *, bs);
   }
   simple_mtx_unlock(&ctx->batch_mtx);
}

void
zink_batch_state_destroy(struct zink_screen *screen, struct zink_batch_state *bs)
{
   if (!bs)
      return;

   util_queue_fence_destroy(&bs->flush_completed);

   if (bs->fence.fence)
      vkDestroyFence(screen->dev, bs->fence.fence, NULL);

   if (bs->cmdbuf)
      vkFreeCommandBuffers(screen->dev, bs->cmdpool, 1, &bs->cmdbuf);
   if (bs->cmdpool)
      vkDestroyCommandPool(screen->dev, bs->cmdpool, NULL);

   if (vkResetDescriptorPool(screen->dev, batch->descpool, 0) != VK_SUCCESS)
      fprintf(stderr, "vkResetDescriptorPool failed\n");
}

static struct zink_batch_state *
create_batch_state(struct zink_context *ctx)
{
   struct zink_screen *screen = zink_screen(ctx->base.screen);
   struct zink_batch_state *bs = rzalloc(NULL, struct zink_batch_state);
   VkCommandPoolCreateInfo cpci = {};
   cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
   cpci.queueFamilyIndex = screen->gfx_queue;
   cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
   if (vkCreateCommandPool(screen->dev, &cpci, NULL, &bs->cmdpool) != VK_SUCCESS)
      goto fail;

   VkCommandBufferAllocateInfo cbai = {};
   cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
   cbai.commandPool = bs->cmdpool;
   cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cbai.commandBufferCount = 1;

   if (vkAllocateCommandBuffers(screen->dev, &cbai, &bs->cmdbuf) != VK_SUCCESS)
      goto fail;

#define SET_CREATE_OR_FAIL(ptr) \
   ptr = _mesa_pointer_set_create(bs); \
   if (!ptr) \
      goto fail

   bs->ctx = ctx;
   pipe_reference_init(&bs->reference, 1);

   SET_CREATE_OR_FAIL(bs->fbs);
   SET_CREATE_OR_FAIL(bs->fence.resources);
   SET_CREATE_OR_FAIL(bs->surfaces);
   SET_CREATE_OR_FAIL(bs->bufferviews);
   SET_CREATE_OR_FAIL(bs->programs);
   SET_CREATE_OR_FAIL(bs->desc_sets);
   SET_CREATE_OR_FAIL(bs->active_queries);
   util_dynarray_init(&bs->zombie_samplers, NULL);
   util_dynarray_init(&bs->persistent_resources, NULL);

   VkFenceCreateInfo fci = {};
   fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

   if (vkCreateFence(screen->dev, &fci, NULL, &bs->fence.fence) != VK_SUCCESS)
      goto fail;

   simple_mtx_init(&bs->fence.resource_mtx, mtx_plain);
   util_queue_fence_init(&bs->flush_completed);

   return bs;
fail:
   zink_batch_state_destroy(screen, bs);
   return NULL;
}

static bool
find_unused_state(struct hash_entry *entry)
{
   struct zink_fence *fence = entry->data;
   /* we can't reset these from fence_finish because threads */
   bool completed = p_atomic_read(&fence->completed);
   bool submitted = p_atomic_read(&fence->submitted);
   return submitted && completed;
}

static struct zink_batch_state *
get_batch_state(struct zink_context *ctx, struct zink_batch *batch)
{
   struct zink_batch_state *bs = NULL;

   simple_mtx_lock(&ctx->batch_mtx);
   if (util_dynarray_num_elements(&ctx->free_batch_states, struct zink_batch_state*))
      bs = util_dynarray_pop(&ctx->free_batch_states, struct zink_batch_state*);
   if (!bs) {
      struct hash_entry *he = _mesa_hash_table_random_entry(&ctx->batch_states, find_unused_state);
      if (he) { //there may not be any entries available
         bs = he->data;
         _mesa_hash_table_remove(&ctx->batch_states, he);
      }
   }
   simple_mtx_unlock(&ctx->batch_mtx);
   if (bs)
      zink_reset_batch_state(ctx, bs);
   else {
      if (!batch->state) {
         /* this is batch init, so create a few more states for later use */
         for (int i = 0; i < 3; i++) {
            struct zink_batch_state *state = create_batch_state(ctx);
            util_dynarray_append(&ctx->free_batch_states, struct zink_batch_state *, state);
         }
      }
      bs = create_batch_state(ctx);
   }
   return bs;
}

void
zink_start_batch(struct zink_context *ctx, struct zink_batch *batch)
{
   reset_batch(ctx, batch);

   VkCommandBufferBeginInfo cbbi = {};
   cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
   cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   if (vkBeginCommandBuffer(batch->cmdbuf, &cbbi) != VK_SUCCESS)
      debug_printf("vkBeginCommandBuffer failed\n");

   batch->state->fence.batch_id = ctx->curr_batch;
   batch->state->fence.completed = false;
   if (ctx->last_fence) {
      struct zink_batch_state *last_state = zink_batch_state(ctx->last_fence);
      batch->last_batch_id = last_state->fence.batch_id;
   } else {
      if (zink_screen(ctx->base.screen)->threaded)
         util_queue_init(&batch->flush_queue, "zfq", 8, 1, UTIL_QUEUE_INIT_RESIZE_IF_FULL);
   }
   if (!ctx->queries_disabled)
      zink_resume_queries(ctx, batch);
}

void
zink_end_batch(struct zink_context *ctx, struct zink_batch *batch)
{
   if (!ctx->queries_disabled)
      zink_suspend_queries(ctx, batch);

   if (vkEndCommandBuffer(batch->cmdbuf) != VK_SUCCESS) {
      debug_printf("vkEndCommandBuffer failed\n");
      return;
   }

   assert(batch->fence == NULL);
   batch->fence = zink_create_fence(ctx->base.screen, batch);
   if (!batch->fence)
      return;

   VkSubmitInfo si = {};
   si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
   si.waitSemaphoreCount = 0;
   si.pWaitSemaphores = NULL;
   si.signalSemaphoreCount = 0;
   si.pSignalSemaphores = NULL;
   si.pWaitDstStageMask = NULL;
   si.commandBufferCount = 1;
   si.pCommandBuffers = &batch->cmdbuf;

   if (vkQueueSubmit(ctx->queue, 1, &si, batch->fence->fence) != VK_SUCCESS) {
      debug_printf("ZINK: vkQueueSubmit() failed\n");
      ctx->is_device_lost = true;

      if (ctx->reset.reset) {
         ctx->reset.reset(ctx->reset.data, PIPE_GUILTY_CONTEXT_RESET);
      }
   }
   vkResetFences(zink_screen(ctx->base.screen)->dev, 1, &batch->state->fence.fence);

   util_dynarray_foreach(&batch->state->persistent_resources, struct zink_resource*, res) {
       struct zink_screen *screen = zink_screen(ctx->base.screen);
       assert(!(*res)->obj->offset);
       VkMappedMemoryRange range = {
          VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          NULL,
          (*res)->obj->mem,
          (*res)->obj->offset,
          VK_WHOLE_SIZE,
       };
       vkFlushMappedMemoryRanges(screen->dev, 1, &range);
   }

   simple_mtx_lock(&ctx->batch_mtx);
   ctx->last_fence = &batch->state->fence;
   _mesa_hash_table_insert_pre_hashed(&ctx->batch_states, batch->state->fence.batch_id, (void*)(uintptr_t)batch->state->fence.batch_id, batch->state);
   simple_mtx_unlock(&ctx->batch_mtx);
   ctx->resource_size += batch->state->resource_size;

   if (util_queue_is_initialized(&batch->flush_queue)) {
      batch->state->queue = batch->thread_queue;
      util_queue_add_job(&batch->flush_queue, batch->state, &batch->state->flush_completed,
                         submit_queue, post_submit, 0);
   } else {
      batch->state->queue = batch->queue;
      submit_queue(batch->state, 0);
      post_submit(batch->state, 0);
   }
}

void
zink_batch_reference_resource_rw(struct zink_batch *batch, struct zink_resource *res, bool write)
{
   unsigned mask = write ? ZINK_RESOURCE_ACCESS_WRITE : ZINK_RESOURCE_ACCESS_READ;

   /* u_transfer_helper unrefs the stencil buffer when the depth buffer is unrefed,
    * so we add an extra ref here to the stencil buffer to compensate
    */
   struct zink_resource *stencil;

   zink_get_depth_stencil_resources((struct pipe_resource*)res, NULL, &stencil);


   struct set_entry *entry = _mesa_set_search(batch->resources, res);
   if (!entry) {
      entry = _mesa_set_add(batch->resources, res);
      pipe_reference(NULL, &res->base.reference);
      if (stencil)
         pipe_reference(NULL, &stencil->base.reference);
   }
   /* the batch_uses value for this batch is guaranteed to not be in use now because
    * reset_batch() waits on the fence and removes access before resetting
    */
   res->batch_uses[batch->batch_id] |= mask;

   if (stencil)
      stencil->batch_uses[batch->batch_id] |= mask;
}

void
zink_batch_reference_sampler_view(struct zink_batch *batch,
                                  struct zink_sampler_view *sv)
{
   struct set_entry *entry = _mesa_set_search(batch->sampler_views, sv);
   if (!entry) {
      entry = _mesa_set_add(batch->sampler_views, sv);
      pipe_reference(NULL, &sv->base.reference);
   }
}

void
zink_batch_reference_program(struct zink_batch *batch,
                             struct zink_gfx_program *prog)
{
   struct set_entry *entry = _mesa_set_search(batch->programs, prog);
   if (!entry) {
      entry = _mesa_set_add(batch->programs, prog);
      pipe_reference(NULL, &prog->reference);
   }
}
