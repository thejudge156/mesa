/*
 * Copyright © 2022 Imagination Technologies Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdint.h>
#include <vulkan/vulkan.h>

#include "hwdef/rogue_hw_defs.h"
#include "pvr_csb.h"
#include "pvr_device_info.h"
#include "pvr_formats.h"
#include "pvr_private.h"
#include "pvr_tex_state.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_log.h"

static enum ROGUE_TEXSTATE_SWIZ pvr_get_hw_swizzle(VkComponentSwizzle comp,
                                                   enum pipe_swizzle swz)
{
   switch (swz) {
   case PIPE_SWIZZLE_0:
      return ROGUE_TEXSTATE_SWIZ_SRC_ZERO;
   case PIPE_SWIZZLE_1:
      return ROGUE_TEXSTATE_SWIZ_SRC_ONE;
   case PIPE_SWIZZLE_X:
      return ROGUE_TEXSTATE_SWIZ_SRCCHAN_0;
   case PIPE_SWIZZLE_Y:
      return ROGUE_TEXSTATE_SWIZ_SRCCHAN_1;
   case PIPE_SWIZZLE_Z:
      return ROGUE_TEXSTATE_SWIZ_SRCCHAN_2;
   case PIPE_SWIZZLE_W:
      return ROGUE_TEXSTATE_SWIZ_SRCCHAN_3;
   case PIPE_SWIZZLE_NONE:
      if (comp == VK_COMPONENT_SWIZZLE_A)
         return ROGUE_TEXSTATE_SWIZ_SRC_ONE;
      else
         return ROGUE_TEXSTATE_SWIZ_SRC_ZERO;
   default:
      unreachable("Unknown enum pipe_swizzle");
   };
}

VkResult
pvr_pack_tex_state(struct pvr_device *device,
                   struct pvr_texture_state_info *info,
                   uint64_t state[static const ROGUE_NUM_TEXSTATE_IMAGE_WORDS])
{
   const struct pvr_device_info *dev_info = &device->pdevice->dev_info;
   uint32_t texture_type;

   pvr_csb_pack (&state[0], TEXSTATE_IMAGE_WORD0, word0) {
      /* Determine texture type */
      if (info->is_cube && info->tex_state_type == PVR_TEXTURE_STATE_SAMPLE) {
         word0.textype = texture_type = PVRX(TEXSTATE_TEXTYPE_CUBE);
      } else if (info->mem_layout == PVR_MEMLAYOUT_TWIDDLED ||
                 info->mem_layout == PVR_MEMLAYOUT_3DTWIDDLED) {
         if (info->type == VK_IMAGE_VIEW_TYPE_3D) {
            word0.textype = texture_type = PVRX(TEXSTATE_TEXTYPE_3D);
         } else if (info->type == VK_IMAGE_VIEW_TYPE_1D ||
                    info->type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
            word0.textype = texture_type = PVRX(TEXSTATE_TEXTYPE_1D);
         } else if (info->type == VK_IMAGE_VIEW_TYPE_2D ||
                    info->type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) {
            word0.textype = texture_type = PVRX(TEXSTATE_TEXTYPE_2D);
         } else {
            return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
         }
      } else if (info->mem_layout == PVR_MEMLAYOUT_LINEAR) {
         word0.textype = texture_type = PVRX(TEXSTATE_TEXTYPE_STRIDE);
      } else {
         return vk_error(device, VK_ERROR_FORMAT_NOT_SUPPORTED);
      }

      word0.texformat = pvr_get_tex_format(info->format);
      word0.smpcnt = util_logbase2(info->sample_count);
      word0.swiz0 =
         pvr_get_hw_swizzle(VK_COMPONENT_SWIZZLE_R, info->swizzle[0]);
      word0.swiz1 =
         pvr_get_hw_swizzle(VK_COMPONENT_SWIZZLE_G, info->swizzle[1]);
      word0.swiz2 =
         pvr_get_hw_swizzle(VK_COMPONENT_SWIZZLE_B, info->swizzle[2]);
      word0.swiz3 =
         pvr_get_hw_swizzle(VK_COMPONENT_SWIZZLE_A, info->swizzle[3]);

      /* Gamma */
      if (vk_format_is_srgb(info->format)) {
         /* Gamma for 2 Component Formats has to be handled differently. */
         if (vk_format_get_nr_components(info->format) == 2) {
            /* Enable Gamma only for Channel 0 if Channel 1 is an Alpha
             * Channel.
             */
            if (vk_format_has_alpha(info->format)) {
               word0.twocomp_gamma = PVRX(TEXSTATE_TWOCOMP_GAMMA_R);
            } else {
               /* Otherwise Enable Gamma for both the Channels. */
               word0.twocomp_gamma = PVRX(TEXSTATE_TWOCOMP_GAMMA_RG);

               /* If Channel 0 happens to be the Alpha Channel, the
                * ALPHA_MSB bit would not be set thereby disabling Gamma
                * for Channel 0.
                */
            }
         } else {
            word0.gamma = PVRX(TEXSTATE_GAMMA_ON);
         }
      }

      word0.width = info->extent.width - 1;
      if (info->type != VK_IMAGE_VIEW_TYPE_1D ||
          info->type != VK_IMAGE_VIEW_TYPE_1D_ARRAY)
         word0.height = info->extent.height - 1;
   }

   /* Texture type specific stuff (word 1) */
   if (texture_type == PVRX(TEXSTATE_TEXTYPE_STRIDE)) {
      pvr_csb_pack (&state[1], TEXSTATE_STRIDE_IMAGE_WORD1, word1) {
         word1.stride = info->stride;
         word1.num_mip_levels = info->mip_levels;
         word1.mipmaps_present = info->mipmaps_present;

         word1.texaddr = info->addr;
         word1.texaddr.addr += info->offset;

         if (vk_format_is_alpha_on_msb(info->format))
            word1.alpha_msb = true;

         if (!PVR_HAS_FEATURE(dev_info, tpu_extended_integer_lookup) &&
             !PVR_HAS_FEATURE(dev_info, tpu_image_state_v2)) {
            if (info->flags & PVR_TEXFLAGS_INDEX_LOOKUP ||
                info->flags & PVR_TEXFLAGS_BUFFER)
               word1.index_lookup = true;
         }

         if (info->flags & PVR_TEXFLAGS_BUFFER)
            word1.mipmaps_present = false;

         if (PVR_HAS_FEATURE(dev_info, tpu_image_state_v2) &&
             vk_format_is_compressed(info->format))
            word1.tpu_image_state_v2_compression_mode =
               PVRX(TEXSTATE_COMPRESSION_MODE_TPU);
      }
   } else {
      pvr_csb_pack (&state[1], TEXSTATE_IMAGE_WORD1, word1) {
         word1.num_mip_levels = info->mip_levels;
         word1.mipmaps_present = info->mipmaps_present;
         word1.baselevel = info->base_level;

         if (info->extent.depth > 0) {
            word1.depth = info->extent.depth - 1;
         } else if (PVR_HAS_FEATURE(dev_info, tpu_array_textures)) {
            uint32_t array_layers = info->array_size;

            if (info->type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY &&
                info->tex_state_type == PVR_TEXTURE_STATE_SAMPLE)
               array_layers /= 6;

            word1.depth = array_layers - 1;
         }

         word1.texaddr = info->addr;
         word1.texaddr.addr += info->offset;

         if (!PVR_HAS_FEATURE(dev_info, tpu_extended_integer_lookup) &&
             !PVR_HAS_FEATURE(dev_info, tpu_image_state_v2)) {
            if (info->flags & PVR_TEXFLAGS_INDEX_LOOKUP ||
                info->flags & PVR_TEXFLAGS_BUFFER)
               word1.index_lookup = true;
         }

         if (info->flags & PVR_TEXFLAGS_BUFFER)
            word1.mipmaps_present = false;

         if (info->flags & PVR_TEXFLAGS_BORDER)
            word1.border = true;

         if (vk_format_is_alpha_on_msb(info->format))
            word1.alpha_msb = true;

         if (PVR_HAS_FEATURE(dev_info, tpu_image_state_v2) &&
             vk_format_is_compressed(info->format))
            word1.tpu_image_state_v2_compression_mode =
               PVRX(TEXSTATE_COMPRESSION_MODE_TPU);
      }
   }

   return VK_SUCCESS;
}