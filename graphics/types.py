from enum import IntEnum
from typing import List, Tuple

import vulkan as vk


class BufferUsage(IntEnum):
    INDEX = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    TRANSFER_DESTINATION = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    VERTEX = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT


class CommandBufferUsage(IntEnum):
    ONE_TIME_SUBMIT = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT


class Format(IntEnum):
    D24X8 = vk.VK_FORMAT_X8_D24_UNORM_PACK32
    R8G8B8A8_UNORM = vk.VK_FORMAT_R8G8B8A8_UNORM
    R8G8B8A8_SRGB = vk.VK_FORMAT_R8G8B8A8_SRGB
    R32G32B32_FLOAT = vk.VK_FORMAT_R32G32B32_SFLOAT
    R32G32B32A32_FLOAT = vk.VK_FORMAT_R32G32B32A32_SFLOAT


class ImageAspect(IntEnum):
    COLOR = vk.VK_IMAGE_ASPECT_COLOR_BIT
    DEPTH = vk.VK_IMAGE_ASPECT_DEPTH_BIT


class ImageLayout(IntEnum):
    COLOR = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    DEPTH = 3  # vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    GENERAL = vk.VK_IMAGE_LAYOUT_GENERAL
    UNDEFINED = vk.VK_IMAGE_LAYOUT_UNDEFINED
    TRANSFER_SOURCE = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    TRANSFER_DESTINATION = vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL


class ImageUsage(IntEnum):
    COLOR = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    DEPTH = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    TRANSFER_DESTINATION = vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
    TRANSFER_SOURCE = vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT


class IndexType(IntEnum):
    UINT16 = vk.VK_INDEX_TYPE_UINT16
    UINT32 = vk.VK_INDEX_TYPE_UINT32


class LoadOp(IntEnum):
    CLEAR = vk.VK_ATTACHMENT_LOAD_OP_CLEAR
    DONT_CARE = vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE
    LOAD = vk.VK_ATTACHMENT_LOAD_OP_LOAD


class StoreOp(IntEnum):
    DISCARD = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE
    STORE = vk.VK_ATTACHMENT_STORE_OP_STORE


class VertexInputRate(IntEnum):
    PER_VERTEX = vk.VK_VERTEX_INPUT_RATE_VERTEX
    PER_INSTANCE = vk.VK_VERTEX_INPUT_RATE_INSTANCE


def AttachmentDescription(
    *,
    format: Format,
    sample_count: int = 1,
    load_op: LoadOp = LoadOp.DONT_CARE,
    store_op: StoreOp = StoreOp.DISCARD,
    initial_layout: ImageLayout = ImageLayout.UNDEFINED,
    final_layout: ImageLayout = ImageLayout.GENERAL,
):
    # pylint: disable=invalid-name
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


def ClearValue(*, color: Tuple[float, float, float, float] = None, depth: float = None):
    # pylint: disable=invalid-name
    if color:
        return vk.VkClearValue(color=vk.VkClearColorValue(float32=color))
    return vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=depth))


def DepthDescription(*, test_enabled: bool = False, write_enabled: bool = False):
    # pylint: disable=invalid-name
    return vk.VkPipelineDepthStencilStateCreateInfo(
        depthTestEnable=test_enabled,
        depthWriteEnable=write_enabled,
        depthCompareOp=vk.VK_COMPARE_OP_LESS,
    )


def GraphicsPipelineDescription(
    *,
    pipeline_layout,
    render_pass,
    vertex_shader,
    fragment_shader,
    vertex_attributes: List = [],
    vertex_bindings: List = [],
    width: int,
    height: int,
    depth_description=None,
):
    # pylint: disable=invalid-name,too-many-locals
    extent = vk.VkExtent2D(width, height)

    shader_stages = [
        vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vertex_shader, pName="main"
        ),
        vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=fragment_shader, pName="main"
        ),
    ]

    vertex_input_state = vk.VkPipelineVertexInputStateCreateInfo(
        pVertexBindingDescriptions=vertex_bindings,
        pVertexAttributeDescriptions=vertex_attributes,
    )

    input_assembly_state = vk.VkPipelineInputAssemblyStateCreateInfo(
        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    )

    viewport_state = vk.VkPipelineViewportStateCreateInfo(
        pViewports=[
            vk.VkViewport(width=width, height=height, minDepth=0.0, maxDepth=1.0)
        ],
        pScissors=[vk.VkRect2D(extent=extent)],
    )

    rasterizer_state = vk.VkPipelineRasterizationStateCreateInfo(
        depthClampEnable=vk.VK_FALSE,
        rasterizerDiscardEnable=vk.VK_FALSE,
        polygonMode=vk.VK_POLYGON_MODE_FILL,
        cullMode=vk.VK_CULL_MODE_BACK_BIT,
        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
        lineWidth=1,
        depthBiasEnable=vk.VK_FALSE,
        depthBiasConstantFactor=0.0,
        depthBiasClamp=0.0,
        depthBiasSlopeFactor=0.0,
    )

    multisample_state = vk.VkPipelineMultisampleStateCreateInfo(
        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
    )

    color_blend_state = vk.VkPipelineColorBlendStateCreateInfo(
        pAttachments=[
            vk.VkPipelineColorBlendAttachmentState(
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
                | vk.VK_COLOR_COMPONENT_G_BIT
                | vk.VK_COLOR_COMPONENT_B_BIT
                | vk.VK_COLOR_COMPONENT_A_BIT
            )
        ]
    )

    context = locals().values()

    return (
        vk.VkGraphicsPipelineCreateInfo(
            pStages=shader_stages,
            pVertexInputState=vertex_input_state,
            pInputAssemblyState=input_assembly_state,
            pTessellationState=None,
            pViewportState=viewport_state,
            pRasterizationState=rasterizer_state,
            pMultisampleState=multisample_state,
            pDepthStencilState=depth_description,
            pColorBlendState=color_blend_state,
            pDynamicState=None,
            layout=pipeline_layout,
            renderPass=render_pass,
            subpass=0,
        ),
        context,
    )


def SubpassDescription(
    *, color_attachments: List[Tuple[int, ImageLayout]], depth_attachment: int = None
):
    # pylint: disable=invalid-name
    if depth_attachment:
        depth_attachment = vk.VkAttachmentReference(
            attachment=1, layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

    color_attachments = [
        vk.VkAttachmentReference(attachment=attachment_index, layout=layout)
        for attachment_index, layout in color_attachments
    ]

    return vk.VkSubpassDescription(
        pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
        pColorAttachments=color_attachments,
        pDepthStencilAttachment=depth_attachment,
    )


def VertexAttribute(*, location: int, binding: int, format: Format, offset: int = 0):
    # pylint: disable=invalid-name
    return vk.VkVertexInputAttributeDescription(
        location=location, binding=binding, format=format, offset=offset
    )


def VertexBinding(
    *,
    binding: int,
    stride: int,
    input_rate: VertexInputRate = VertexInputRate.PER_VERTEX,
):
    # pylint: disable=invalid-name
    return vk.VkVertexInputBindingDescription(
        binding=binding, stride=stride, inputRate=input_rate
    )
