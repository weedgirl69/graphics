from enum import IntEnum
from typing import List, Tuple

import vulkan as vk


class CommandBufferUsage(IntEnum):
    ONE_TIME_SUBMIT = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT


class Format(IntEnum):
    R8G8B8A8_UNORM = vk.VK_FORMAT_R8G8B8A8_UNORM
    R8G8B8A8_SRGB = vk.VK_FORMAT_R8G8B8A8_SRGB


class LoadOp(IntEnum):
    CLEAR = vk.VK_ATTACHMENT_LOAD_OP_CLEAR
    DONT_CARE = vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE
    LOAD = vk.VK_ATTACHMENT_LOAD_OP_LOAD


class StoreOp(IntEnum):
    DISCARD = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE
    STORE = vk.VK_ATTACHMENT_STORE_OP_STORE


class ImageLayout(IntEnum):
    COLOR = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    UNDEFINED = vk.VK_IMAGE_LAYOUT_UNDEFINED
    TRANSFER_SOURCE = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    TRANSFER_DESTINATION = vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL


def attachment_description(
    format: Format,
    sample_count: int = 1,
    load_op: LoadOp = LoadOp.DONT_CARE,
    store_op: StoreOp = StoreOp.DISCARD,
    initial_layout: ImageLayout = ImageLayout.UNDEFINED,
    final_layout: ImageLayout = ImageLayout.UNDEFINED,
):
    return vk.VkAttachmentDescription(
        format=format,
        samples=1 << (sample_count - 1),
        loadOp=load_op,
        storeOp=store_op,
        stencilLoadOp=LoadOp.DONT_CARE,
        stencilStoreOp=StoreOp.DISCARD,
        initialLayout=initial_layout,
        finalLayout=final_layout,
    )


def subpass_description(attachments: List[Tuple[int, ImageLayout]]):
    return vk.VkSubpassDescription(
        pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
        pColorAttachments=[
            vk.VkAttachmentReference(
                attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            )
            for attachment_index, layout in attachments
        ],
    )
