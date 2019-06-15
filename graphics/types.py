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
    R8G8B8A8_UNORM = vk.VK_FORMAT_R8G8B8A8_UNORM
    R8G8B8A8_SRGB = vk.VK_FORMAT_R8G8B8A8_SRGB
    R32G32B32_FLOAT = vk.VK_FORMAT_R32G32B32_SFLOAT


class ImageLayout(IntEnum):
    COLOR = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    UNDEFINED = vk.VK_IMAGE_LAYOUT_UNDEFINED
    TRANSFER_SOURCE = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    TRANSFER_DESTINATION = vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL


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


def attachment_description(
    *,
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


class GraphicsPipelineDescription:
    def __init__(
        self,
        *,
        pipeline_layout,
        render_pass,
        vertex_shader,
        fragment_shader,
        vertex_attributes: List = [],
        vertex_bindings: List = [],
        width: int,
        height: int,
    ):
        extent = vk.VkExtent2D(width, height)

        self.shader_stages = [
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vertex_shader, pName="main"
            ),
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=fragment_shader,
                pName="main",
            ),
        ]

        self.vertex_input_state = vk.VkPipelineVertexInputStateCreateInfo(
            pVertexBindingDescriptions=vertex_bindings,
            pVertexAttributeDescriptions=vertex_attributes,
        )

        self.input_assembly_state = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )

        self.viewport_state = vk.VkPipelineViewportStateCreateInfo(
            pViewports=[
                vk.VkViewport(width=width, height=height, minDepth=0.0, maxDepth=1.0)
            ],
            pScissors=[vk.VkRect2D(extent=extent)],
        )

        self.rasterizer_state = vk.VkPipelineRasterizationStateCreateInfo(
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

        self.multisample_state = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
        )

        self.color_blend_state = vk.VkPipelineColorBlendStateCreateInfo(
            pAttachments=[
                vk.VkPipelineColorBlendAttachmentState(
                    colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
                    | vk.VK_COLOR_COMPONENT_G_BIT
                    | vk.VK_COLOR_COMPONENT_B_BIT
                    | vk.VK_COLOR_COMPONENT_A_BIT
                )
            ]
        )

        self.create_info = vk.VkGraphicsPipelineCreateInfo(
            pStages=self.shader_stages,
            pVertexInputState=self.vertex_input_state,
            pInputAssemblyState=self.input_assembly_state,
            pTessellationState=None,
            pViewportState=self.viewport_state,
            pRasterizationState=self.rasterizer_state,
            pMultisampleState=self.multisample_state,
            pDepthStencilState=None,
            pColorBlendState=self.color_blend_state,
            pDynamicState=None,
            layout=pipeline_layout,
            renderPass=render_pass,
            subpass=0,
        )


def vertex_attribute(*, location: int, binding: int, format: Format, offset: int = 0):
    return vk.VkVertexInputAttributeDescription(
        location=location, binding=binding, format=format, offset=offset
    )


def vertex_binding(
    *,
    binding: int,
    stride: int,
    input_rate: VertexInputRate = VertexInputRate.PER_VERTEX,
):
    return vk.VkVertexInputBindingDescription(
        binding=binding, stride=stride, inputRate=input_rate
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
