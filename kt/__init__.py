import dataclasses
from enum import Enum, IntEnum
import typing
from typing import List, NewType, Optional, Tuple
import vulkan as vk

Buffer = NewType("Buffer", object)
ClearValue = NewType("ClearValue", object)
CommandBuffer = NewType("CommandBuffer", object)
CommandPool = NewType("CommandPool", object)
DescriptorPool = NewType("DescriptorPool", object)
DescriptorSet = NewType("DescriptorSet", object)
DescriptorSetLayout = NewType("DescriptorSetLayout", object)
Device = NewType("Device", object)
DeviceMemory = NewType("DeviceMemory", object)
Framebuffer = NewType("Framebuffer", object)
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
WriteDescriptorImage = NewType("WriteDescriptorImage", object)


class BufferUsage(IntEnum):
    INDEX = vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    TRANSFER_DESTINATION = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    TRANSFER_SOURCE = vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    UNIFORM = vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    VERTEX = vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT


class CommandBufferUsage(IntEnum):
    ONE_TIME_SUBMIT = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT


class DescriptorType(IntEnum):
    COMBINED_IMAGE_SAMPLER = vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
    UNIFORM_BUFFER = vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER


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


class ShaderStage(IntEnum):
    VERTEX = vk.VK_SHADER_STAGE_VERTEX_BIT
    FRAGMENT = vk.VK_SHADER_STAGE_FRAGMENT_BIT


class StoreOp(IntEnum):
    DISCARD = vk.VK_ATTACHMENT_STORE_OP_DONT_CARE
    STORE = vk.VK_ATTACHMENT_STORE_OP_STORE


class VertexInputRate(IntEnum):
    PER_VERTEX = vk.VK_VERTEX_INPUT_RATE_VERTEX
    PER_INSTANCE = vk.VK_VERTEX_INPUT_RATE_INSTANCE


@dataclasses.dataclass(frozen=True)
class AttachmentDescription:
    pixel_format: Format
    sample_count: int = 0
    load_op: LoadOp = LoadOp.DONT_CARE
    store_op: StoreOp = StoreOp.DISCARD
    initial_layout: ImageLayout = ImageLayout.UNDEFINED
    final_layout: ImageLayout = ImageLayout.GENERAL


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


@dataclasses.dataclass(frozen=True)
class DepthDescription:
    test_enabled: bool = False
    write_enabled: bool = False


@dataclasses.dataclass(frozen=True)
class DescriptorSetLayoutBinding:
    stage: ShaderStage
    descriptor_type: DescriptorType
    count: int = 1
    immutable_samplers: typing.Optional[typing.List[Sampler]] = None


@dataclasses.dataclass(frozen=True)
class VertexAttribute:
    binding: int
    pixel_format: Format
    offset: int = 0


@dataclasses.dataclass(frozen=True)
class VertexBinding:
    stride: int
    input_rate: VertexInputRate = VertexInputRate.PER_VERTEX


@dataclasses.dataclass(frozen=True)
class GraphicsPipelineDescription:
    pipeline_layout: PipelineLayout
    render_pass: RenderPass
    vertex_shader: ShaderModule
    fragment_shader: ShaderModule
    width: int
    height: int
    vertex_attributes: Optional[List[VertexAttribute]] = None
    vertex_bindings: Optional[List[VertexBinding]] = None
    sample_count: int = 0
    depth_description: Optional[DepthDescription] = None


@dataclasses.dataclass(frozen=True)
class SubpassDescription:
    color_attachments: typing.List[typing.Tuple[int, ImageLayout]]
    depth_attachment_index: typing.Optional[int] = None
    resolve_attachments: typing.Optional[
        typing.List[typing.Tuple[int, ImageLayout]]
    ] = None


@dataclasses.dataclass(frozen=True)
class DescriptorBufferInfo:
    buffer: Buffer
    byte_count: int
    byte_offset: int


@dataclasses.dataclass(frozen=True)
class DescriptorImageInfo:
    image_view: ImageView
    layout: ImageLayout


@dataclasses.dataclass(frozen=True)
class DescriptorBufferWrites:
    binding: int
    buffer_infos: typing.List[DescriptorBufferInfo]
    descriptor_set: DescriptorSet
    descriptor_type: DescriptorType
    count: int = 1


@dataclasses.dataclass(frozen=True)
class DescriptorImageWrites:
    binding: int
    count: int
    descriptor_set: DescriptorSet
    descriptor_type: DescriptorType
    image_infos: typing.List[DescriptorImageInfo]
