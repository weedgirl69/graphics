import collections
from enum import Enum, IntEnum
from typing import List, NewType, Optional, Tuple
import vulkan as vk

AttachmentDescription = NewType("AttachmentDescription", object)
Buffer = NewType("Buffer", object)
ClearValue = NewType("ClearValue", object)
CommandBuffer = NewType("CommandBuffer", object)
CommandPool = NewType("CommandPool", object)
DepthDescription = NewType("DepthDescription", object)
DescriptorPool = NewType("DescriptorPool", object)
DescriptorSet = NewType("DescriptorSet", object)
DescriptorSetLayout = NewType("DescriptorSetLayout", object)
Device = NewType("Device", object)
DeviceMemory = NewType("DeviceMemory", object)
Framebuffer = NewType("Framebuffer", object)
GraphicsPipelineDescription = NewType("GraphicsPipelineDescription", object)
Image = NewType("Image", object)
ImageView = NewType("ImageView", object)
Instance = NewType("Instance", object)
MultisampleDescription = NewType("MultisampleDescription", object)
PhysicalDevice = NewType("PhysicalDevice", object)
Pipeline = NewType("Pipeline", object)
PipelineLayout = NewType("PipelineLayout", object)
RenderPass = NewType("RenderPass", object)
Sampler = NewType("Sampler", object)
ShaderModule = NewType("ShaderModule", object)
SubpassDescription = NewType("SubpassDescription", object)
VertexAttribute = NewType("VertexAttribute", object)
VertexBinding = NewType("VertexBinding", object)
WriteDescriptorImage = NewType("WriteDescriptorImage", object)


class BufferUsage(IntEnum):
    INDEX = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    TRANSFER_DESTINATION = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    TRANSFER_SOURCE = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    VERTEX = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT


class CommandBufferUsage(IntEnum):
    ONE_TIME_SUBMIT = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT


class DescriptorType(IntEnum):
    COMBINED_IMAGE_SAMPLER = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER


class Filter(IntEnum):
    NEAREST = vk.VK_FILTER_NEAREST
    LINEAR = vk.VK_FILTER_LINEAR


class Format(IntEnum):
    D24X8 = vk.VK_FORMAT_X8_D24_UNORM_PACK32
    R8G8B8A8_UNORM = vk.VK_FORMAT_R8G8B8A8_UNORM
    R8G8B8A8_SRGB = vk.VK_FORMAT_R8G8B8A8_SRGB
    R32G32_FLOAT = vk.VK_FORMAT_R32G32_SFLOAT
    R32G32B32_FLOAT = vk.VK_FORMAT_R32G32B32_SFLOAT
    R32G32B32A32_FLOAT = vk.VK_FORMAT_R32G32B32A32_SFLOAT


class ImageAspect(IntEnum):
    COLOR = vk.VK_IMAGE_ASPECT_COLOR_BIT
    DEPTH = vk.VK_IMAGE_ASPECT_DEPTH_BIT


class ImageLayout(IntEnum):
    COLOR = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    DEPTH = vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    GENERAL = vk.VK_IMAGE_LAYOUT_GENERAL
    SHADER = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    TRANSFER_SOURCE = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    TRANSFER_DESTINATION = vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    UNDEFINED = vk.VK_IMAGE_LAYOUT_UNDEFINED


class ImageUsage(IntEnum):
    COLOR_ATTACHMENT = vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    DEPTH_ATTACHMENT = vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    SAMPLED = vk.VK_IMAGE_USAGE_SAMPLED_BIT
    TRANSFER_DESTINATION = vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
    TRANSFER_SOURCE = vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT


class IndexType(IntEnum):
    UINT16 = vk.VK_INDEX_TYPE_UINT16
    UINT32 = vk.VK_INDEX_TYPE_UINT32


class LoadOp(IntEnum):
    CLEAR = vk.VK_ATTACHMENT_LOAD_OP_CLEAR
    DONT_CARE = vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE
    LOAD = vk.VK_ATTACHMENT_LOAD_OP_LOAD


class MemoryType(Enum):
    DeviceOptimal = 0
    Uploadable = 1
    Downloadable = 2
    LazilyAllocated = 3


class StoreOp(IntEnum):
    DISCARD = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE
    STORE = vk.VK_ATTACHMENT_STORE_OP_STORE


class VertexInputRate(IntEnum):
    PER_VERTEX = vk.VK_VERTEX_INPUT_RATE_VERTEX
    PER_INSTANCE = vk.VK_VERTEX_INPUT_RATE_INSTANCE


def new_attachment_description(
    *,
    format: Format,
    sample_count: int = 0,
    load_op: LoadOp = LoadOp.DONT_CARE,
    store_op: StoreOp = StoreOp.DISCARD,
    initial_layout: ImageLayout = ImageLayout.UNDEFINED,
    final_layout: ImageLayout = ImageLayout.GENERAL,
) -> AttachmentDescription:
    return AttachmentDescription(
        vk.VkAttachmentDescription(
            format=format,
            samples=1 << sample_count,
            loadOp=load_op,
            storeOp=store_op,
            stencilLoadOp=LoadOp.DONT_CARE,
            stencilStoreOp=StoreOp.DISCARD,
            initialLayout=initial_layout,
            finalLayout=final_layout,
        )
    )


def new_clear_value(
    *,
    color: Optional[Tuple[float, float, float, float]] = None,
    depth: Optional[float] = None,
) -> ClearValue:
    if color:
        return ClearValue(vk.VkClearValue(color=vk.VkClearColorValue(float32=color)))
    return ClearValue(
        vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=depth))
    )


def new_depth_description(
    *, test_enabled: bool = False, write_enabled: bool = False
) -> DepthDescription:
    return DepthDescription(
        vk.VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=test_enabled,
            depthWriteEnable=write_enabled,
            depthCompareOp=vk.VK_COMPARE_OP_LESS,
        )
    )


def new_graphics_pipeline_description(
    *,
    pipeline_layout: PipelineLayout,
    render_pass: RenderPass,
    vertex_shader: ShaderModule,
    fragment_shader: ShaderModule,
    vertex_attributes: Optional[List[VertexAttribute]] = None,
    vertex_bindings: Optional[List[VertexBinding]] = None,
    width: int,
    height: int,
    multisample_description: Optional[MultisampleDescription] = None,
    depth_description: Optional[DepthDescription] = None,
) -> Tuple[GraphicsPipelineDescription, collections.abc.ValuesView]:
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

    if not multisample_description:
        multisample_description = vk.VkPipelineMultisampleStateCreateInfo(
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
            pMultisampleState=multisample_description,
            pDepthStencilState=depth_description,
            pColorBlendState=color_blend_state,
            pDynamicState=None,
            layout=pipeline_layout,
            renderPass=render_pass,
            subpass=0,
        ),
        context,
    )


def new_multisample_description(sample_count: int = 0) -> MultisampleDescription:
    return MultisampleDescription(
        vk.VkPipelineMultisampleStateCreateInfo(rasterizationSamples=1 << sample_count)
    )


def new_subpass_description(
    *,
    color_attachments: List[Tuple[int, ImageLayout]],
    resolve_attachments: Optional[List[Tuple[int, ImageLayout]]] = None,
    depth_attachment: int = None,
) -> SubpassDescription:
    if depth_attachment:
        depth_attachment = vk.VkAttachmentReference(
            attachment=1, layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

    if not resolve_attachments:
        resolve_attachments = []

    color_attachments = [
        vk.VkAttachmentReference(attachment=attachment_index, layout=layout)
        for attachment_index, layout in color_attachments
    ]

    resolve_attachments = [
        vk.VkAttachmentReference(attachment=attachment_index, layout=layout)
        for attachment_index, layout in resolve_attachments
    ]

    return SubpassDescription(
        vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            pColorAttachments=color_attachments,
            pResolveAttachments=resolve_attachments,
            pDepthStencilAttachment=depth_attachment,
        )
    )


def new_vertex_attribute(
    *, location: int, binding: int, format: Format, offset: int = 0
) -> VertexAttribute:
    return VertexAttribute(
        vk.VkVertexInputAttributeDescription(
            location=location, binding=binding, format=format, offset=offset
        )
    )


def new_vertex_binding(
    *,
    binding: int,
    stride: int,
    input_rate: VertexInputRate = VertexInputRate.PER_VERTEX,
) -> VertexBinding:
    return VertexBinding(
        vk.VkVertexInputBindingDescription(
            binding=binding, stride=stride, inputRate=input_rate
        )
    )


def new_write_descriptor_image(
    *,
    descriptor_set: DescriptorSet,
    binding: int,
    image_views_and_layouts: List[Tuple[ImageView, ImageLayout]],
) -> WriteDescriptorImage:
    return WriteDescriptorImage(
        vk.VkWriteDescriptorSet(
            dstSet=descriptor_set,
            dstBinding=binding,
            descriptorType=DescriptorType.COMBINED_IMAGE_SAMPLER,
            pImageInfo=[
                vk.VkDescriptorImageInfo(imageView=image_view, imageLayout=image_layout)
                for image_view, image_layout in image_views_and_layouts
            ],
        )
    )

