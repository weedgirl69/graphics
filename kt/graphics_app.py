import collections
import contextlib
import dataclasses
import itertools
import typing
import functools
import types
import vulkan as vk
import shaderc
import kt


INSTANCE_EXTENSIONS = ["VK_EXT_debug_utils"]
INSTANCE_LAYERS = ["VK_LAYER_LUNARG_standard_validation"]
DEVICE_EXTENSIONS = ["VK_KHR_bind_memory2"]

BUFFER_TYPE = vk.ffi.typeof("struct VkBuffer_T *")
IMAGE_TYPE = vk.ffi.typeof("struct VkImage_T *")

VulkanResourceType = typing.TypeVar("VulkanResourceType")

VulkanResource = typing.Union[
    kt.Buffer,
    kt.CommandPool,
    kt.DescriptorPool,
    kt.DescriptorSetLayout,
    kt.Framebuffer,
    kt.Image,
    kt.ImageView,
    kt.Pipeline,
    kt.PipelineLayout,
    kt.RenderPass,
    kt.Sampler,
    kt.ShaderModule,
]


class Queue:
    def __init__(self, queue: typing.Any) -> None:
        self.queue = queue

    def submit(self, command_buffer: kt.CommandBuffer) -> None:
        vk.vkQueueSubmit(
            self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[command_buffer])], None
        )

    def wait(self) -> None:
        vk.vkQueueWaitIdle(self.queue)


@dataclasses.dataclass(frozen=True)
class VulkanContext:
    instance: kt.Instance
    physical_device: kt.PhysicalDevice
    graphics_queue_family_index: int
    graphics_queue: Queue
    device: kt.Device
    vk_bind_buffer_memory2: typing.Callable
    vk_bind_image_memory2: typing.Callable
    memory_types: typing.Dict[kt.MemoryType, int]
    errors: typing.List[str]
    allocations: typing.Dict[object, kt.DeviceMemory]
    resources: typing.List[VulkanResource]


def _add_to_resources(method: typing.Callable) -> typing.Callable:
    @functools.wraps(method)
    def wrapper(
        self: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> VulkanResourceType:
        result = method(self, *args, **kwargs)
        if isinstance(result, typing.List):
            self.context.resources.extend(result)
        else:
            self.context.resources.append(result)
        return result

    return wrapper


class GraphicsApp:
    def __init__(self, context: VulkanContext) -> None:
        self.context: VulkanContext = context
        self.errors = context.errors
        self.graphics_queue = context.graphics_queue

    def allocate_command_buffer(self, command_pool: kt.CommandPool) -> kt.CommandBuffer:
        command_buffers = vk.vkAllocateCommandBuffers(
            self.context.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=command_pool, commandBufferCount=1
            ),
        )
        return kt.CommandBuffer(command_buffers[0])

    def allocate_descriptor_sets(
        self,
        *,
        descriptor_pool: kt.DescriptorPool,
        descriptor_set_layouts: typing.List[kt.DescriptorSetLayout] = [],
    ) -> typing.List[kt.DescriptorSet]:
        return [
            kt.DescriptorSet(descriptor_set)
            for descriptor_set in vk.vkAllocateDescriptorSets(
                self.context.device,
                vk.VkDescriptorSetAllocateInfo(
                    descriptorPool=descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=descriptor_set_layouts,
                ),
            )
        ]

    @_add_to_resources
    def new_buffer(self, *, byte_count: int, usage: kt.BufferUsage) -> kt.Buffer:
        return kt.Buffer(
            vk.vkCreateBuffer(
                self.context.device,
                vk.VkBufferCreateInfo(size=byte_count, usage=usage),
                None,
            )
        )

    @_add_to_resources
    def new_command_pool(self) -> kt.CommandPool:
        return kt.CommandPool(
            vk.vkCreateCommandPool(
                self.context.device,
                vk.VkCommandPoolCreateInfo(
                    flags=vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                    queueFamilyIndex=self.context.graphics_queue_family_index,
                ),
                None,
            )
        )

    @_add_to_resources
    def new_descriptor_pool(
        self,
        *,
        max_set_count: int,
        descriptor_type_counts: typing.Dict[kt.DescriptorType, int],
    ) -> kt.DescriptorPool:
        pool_sizes = [
            vk.VkDescriptorPoolSize(type=descriptor_type, descriptorCount=count)
            for descriptor_type, count in descriptor_type_counts.items()
        ]
        return kt.DescriptorPool(
            vk.vkCreateDescriptorPool(
                self.context.device,
                vk.VkDescriptorPoolCreateInfo(
                    maxSets=max_set_count, pPoolSizes=pool_sizes
                ),
                None,
            )
        )

    @_add_to_resources
    def new_descriptor_set_layout(
        self, bindings: typing.List[kt.DescriptorSetLayoutBinding] = None
    ) -> kt.DescriptorSetLayout:
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=binding_index,
                descriptorType=binding.descriptor_type,
                descriptorCount=binding.count,
                stageFlags=binding.stage,
                pImmutableSamplers=binding.immutable_samplers,
            )
            for binding_index, binding in enumerate(bindings or [])
        ]
        return kt.DescriptorSetLayout(
            vk.vkCreateDescriptorSetLayout(
                self.context.device,
                vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings),
                None,
            )
        )

    @_add_to_resources
    def new_framebuffer(
        self,
        *,
        render_pass: kt.RenderPass,
        attachments: typing.List[object],
        width: int,
        height: int,
        layers: int = 1,
    ) -> kt.Framebuffer:
        return kt.Framebuffer(
            vk.vkCreateFramebuffer(
                self.context.device,
                vk.VkFramebufferCreateInfo(
                    renderPass=render_pass,
                    pAttachments=attachments,
                    width=width,
                    height=height,
                    layers=layers,
                ),
                None,
            )
        )

    @_add_to_resources
    def new_image(
        self,
        *,
        format: kt.Format,
        usage: kt.ImageUsage,
        width: int,
        height: int,
        mip_count: int = 1,
        sample_count: int = 0,
    ) -> kt.Image:
        return kt.Image(
            vk.vkCreateImage(
                self.context.device,
                vk.VkImageCreateInfo(
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=format,
                    extent=vk.VkExtent3D(width, height, 1),
                    mipLevels=mip_count,
                    arrayLayers=1,
                    usage=usage,
                    samples=1 << sample_count,
                ),
                None,
            )
        )

    @_add_to_resources
    def new_image_view(
        self,
        *,
        image: kt.Image,
        format: kt.Format,
        mip_count: int = 1,
        aspect: kt.ImageAspect = kt.ImageAspect.COLOR,
    ) -> kt.ImageView:
        return kt.ImageView(
            vk.vkCreateImageView(
                self.context.device,
                vk.VkImageViewCreateInfo(
                    image=image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=format,
                    components=vk.VkComponentMapping(
                        vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    ),
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=aspect, levelCount=mip_count, layerCount=1
                    ),
                ),
                None,
            )
        )

    def new_memory_set(
        self,
        *,
        device_optimal: typing.Optional[
            typing.List[typing.Union[kt.Buffer, kt.Image]]
        ] = None,
        downloadable: typing.Optional[
            typing.List[typing.Union[kt.Buffer, kt.Image]]
        ] = None,
        lazily_allocated: typing.Optional[
            typing.List[typing.Union[kt.Buffer, kt.Image]]
        ] = None,
        uploadable: typing.Optional[
            typing.List[typing.Union[kt.Buffer, kt.Image]]
        ] = None,
        initial_values: typing.Dict[typing.Union[kt.Buffer, kt.Image], bytes] = {},
    ) -> typing.Dict[typing.Union[kt.Buffer, kt.Image], vk.ffi.buffer]:
        def select_memory_type_index(
            memory_requirements: typing.Any, memory_type: kt.MemoryType
        ) -> int:
            allowable_memory_type_indices = (
                memory_requirements.memoryTypeBits
                & self.context.memory_types[memory_type]
            )
            assert allowable_memory_type_indices

            bits = bin(allowable_memory_type_indices)
            return len(bits) - len(bits.rstrip("0"))

        memory_type_to_resource_list = {
            kt.MemoryType.DeviceOptimal: device_optimal,
            kt.MemoryType.Downloadable: downloadable,
            kt.MemoryType.LazilyAllocated: lazily_allocated,
            kt.MemoryType.Uploadable: uploadable,
        }

        all_resources = list(
            itertools.chain.from_iterable(
                resource_list
                for resource_list in memory_type_to_resource_list.values()
                if resource_list is not None
            )
        )

        resource_to_memory_requirements = {
            resource: {
                BUFFER_TYPE: vk.vkGetBufferMemoryRequirements,
                IMAGE_TYPE: vk.vkGetImageMemoryRequirements,
            }[vk.ffi.typeof(resource)](self.context.device, resource)
            for resource in all_resources
        }

        resource_to_memory_type_index = {
            resource: select_memory_type_index(
                resource_to_memory_requirements[resource], memory_type
            )
            for memory_type, resource_list in memory_type_to_resource_list.items()
            if resource_list
            for resource in resource_list
        }

        memory_type_index_to_resources: typing.DefaultDict[
            int, typing.List[typing.Union[kt.Buffer, kt.Image]]
        ] = collections.defaultdict(list)
        for resource, memory_type_index in resource_to_memory_type_index.items():
            memory_type_index_to_resources[memory_type_index].append(resource)

        memory_type_index_to_byte_count = {
            memory_type_index: sum(
                (
                    resource_to_memory_requirements[resource].size
                    for resource in resources
                )
            )
            for memory_type_index, resources in memory_type_index_to_resources.items()
        }

        resource_to_byte_offset = {
            resource: byte_offset
            for memory_type_index, resources in memory_type_index_to_resources.items()
            for resource, byte_offset in zip(
                resources,
                [0]
                + list(
                    itertools.accumulate(
                        resource_to_memory_requirements[resource].size
                        for resource in resources
                    )
                )[:-1],
            )
        }

        memory_type_index_to_memory = {
            memory_type_index: vk.vkAllocateMemory(
                self.context.device,
                vk.VkMemoryAllocateInfo(
                    allocationSize=memory_type_index_to_byte_count[memory_type_index],
                    memoryTypeIndex=memory_type_index,
                ),
                None,
            )
            for memory_type_index, resources in memory_type_index_to_resources.items()
        }

        mappable_memory_types = (
            self.context.memory_types[kt.MemoryType.Downloadable]
            | self.context.memory_types[kt.MemoryType.Uploadable]
        )

        memory_type_index_to_mapped_memory = {
            memory_type_index: vk.ffi.from_buffer(
                vk.vkMapMemory(
                    self.context.device,
                    memory_type_index_to_memory[memory_type_index],
                    0,
                    byte_count,
                    0,
                )
            )
            for memory_type_index, byte_count in memory_type_index_to_byte_count.items()
            if (1 << memory_type_index) & mappable_memory_types
        }

        buffers = [
            resource
            for resource in all_resources
            if vk.ffi.typeof(resource) == BUFFER_TYPE
        ]
        images = [
            resource
            for resource in all_resources
            if vk.ffi.typeof(resource) == IMAGE_TYPE
        ]

        if buffers:
            self.context.vk_bind_buffer_memory2(
                self.context.device,
                len(buffers),
                [
                    vk.VkBindBufferMemoryInfo(
                        buffer=buffer,
                        memory=memory_type_index_to_memory[
                            resource_to_memory_type_index[buffer]
                        ],
                        memoryOffset=resource_to_byte_offset[buffer],
                    )
                    for buffer in buffers
                ],
            )

        if images:
            self.context.vk_bind_image_memory2(
                self.context.device,
                len(images),
                [
                    vk.VkBindImageMemoryInfo(
                        image=image,
                        memory=memory_type_index_to_memory[
                            resource_to_memory_type_index[image]
                        ],
                        memoryOffset=resource_to_byte_offset[image],
                    )
                    for image in images
                ],
            )

        mappings = {
            resource: vk.ffi.buffer(
                memory_type_index_to_mapped_memory[memory_type_index]
                + resource_to_byte_offset[resource],
                resource_to_memory_requirements[resource].size,
            )
            for memory_type_index, resources in memory_type_index_to_resources.items()
            if memory_type_index in memory_type_index_to_mapped_memory.keys()
            for resource in resources
        }

        for resource, value in initial_values.items():
            mappings[resource][: len(value)] = value

        mappings = types.MappingProxyType(mappings)

        self.context.allocations[
            frozenset(mappings.keys())
        ] = memory_type_index_to_memory.values()

        return mappings

    def delete_memory_set(
        self, memory_set: typing.Dict[typing.Union[kt.Buffer, kt.Image], vk.ffi.buffer]
    ) -> None:
        memory_key = frozenset(memory_set.keys())
        for memory in self.context.allocations[memory_key]:
            vk.vkFreeMemory(self.context.device, memory, None)

        del self.context.allocations[memory_key]

    @_add_to_resources
    def new_graphics_pipelines(
        self, pipeline_descriptions: typing.List[kt.GraphicsPipelineDescription]
    ) -> typing.List[kt.Pipeline]:
        input_assembly_state = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        rasterization_state = vk.VkPipelineRasterizationStateCreateInfo(
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

        native_shader_stage_lists = [
            [
                vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                    module=pipeline_description.vertex_shader,
                    pName="main",
                ),
                vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    module=pipeline_description.fragment_shader,
                    pName="main",
                ),
            ]
            for pipeline_description in pipeline_descriptions
        ]

        default_depth_state = vk.VkPipelineDepthStencilStateCreateInfo(
            depthCompareOp=vk.VK_COMPARE_OP_LESS
        )

        pipelines = vk.vkCreateGraphicsPipelines(
            self.context.device,
            None,
            len(pipeline_descriptions),
            [
                vk.VkGraphicsPipelineCreateInfo(
                    pColorBlendState=color_blend_state,
                    pDepthStencilState=vk.VkPipelineDepthStencilStateCreateInfo(
                        depthTestEnable=pipeline_description.depth_description.test_enabled,
                        depthWriteEnable=pipeline_description.depth_description.write_enabled,
                        depthCompareOp=vk.VK_COMPARE_OP_LESS,
                    )
                    if pipeline_description.depth_description
                    else default_depth_state,
                    pInputAssemblyState=input_assembly_state,
                    layout=pipeline_description.pipeline_layout,
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        rasterizationSamples=1 << pipeline_description.sample_count
                    ),
                    pRasterizationState=rasterization_state,
                    renderPass=pipeline_description.render_pass,
                    pStages=native_shader_stage_list,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        pVertexBindingDescriptions=[
                            vk.VkVertexInputBindingDescription(
                                binding=binding_index,
                                stride=vertex_binding.stride,
                                inputRate=vertex_binding.input_rate,
                            )
                            for binding_index, vertex_binding in enumerate(
                                pipeline_description.vertex_bindings or []
                            )
                        ],
                        pVertexAttributeDescriptions=[
                            vk.VkVertexInputAttributeDescription(
                                location=location_index,
                                binding=vertex_attribute.binding,
                                format=vertex_attribute.pixel_format,
                                offset=vertex_attribute.offset,
                            )
                            for location_index, vertex_attribute in enumerate(
                                pipeline_description.vertex_attributes or []
                            )
                        ],
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        pViewports=[
                            vk.VkViewport(
                                width=pipeline_description.width,
                                height=pipeline_description.height,
                                minDepth=0.0,
                                maxDepth=1.0,
                            )
                        ],
                        pScissors=[
                            vk.VkRect2D(
                                extent=vk.VkExtent2D(
                                    pipeline_description.width,
                                    pipeline_description.height,
                                )
                            )
                        ],
                    ),
                )
                for native_shader_stage_list, pipeline_description in zip(
                    native_shader_stage_lists, pipeline_descriptions
                )
            ],
            None,
        )

        return [pipelines[i] for i in range(len(pipeline_descriptions))]

    @_add_to_resources
    def new_pipeline_layout(
        self,
        descriptor_set_layouts: typing.Optional[
            typing.List[kt.DescriptorSetLayout]
        ] = None,
    ) -> kt.PipelineLayout:
        return kt.PipelineLayout(
            vk.vkCreatePipelineLayout(
                self.context.device,
                vk.VkPipelineLayoutCreateInfo(pSetLayouts=descriptor_set_layouts),
                None,
            )
        )

    @_add_to_resources
    def new_render_pass(
        self,
        *,
        attachment_descriptions: typing.List[kt.AttachmentDescription],
        subpass_descriptions: typing.List[kt.SubpassDescription],
    ) -> kt.RenderPass:
        attachments = [
            vk.VkAttachmentDescription(
                format=attachment_description.pixel_format,
                samples=1 << attachment_description.sample_count,
                loadOp=attachment_description.load_op,
                storeOp=attachment_description.store_op,
                stencilLoadOp=kt.LoadOp.DONT_CARE,
                stencilStoreOp=kt.StoreOp.DISCARD,
                initialLayout=attachment_description.initial_layout,
                finalLayout=attachment_description.final_layout,
            )
            for attachment_description in attachment_descriptions
        ]
        native_resources = [
            (
                [
                    vk.VkAttachmentReference(attachment=attachment_index, layout=layout)
                    for attachment_index, layout in subpass_description.color_attachments
                ],
                [
                    vk.VkAttachmentReference(attachment=attachment_index, layout=layout)
                    for attachment_index, layout in subpass_description.resolve_attachments
                ]
                if subpass_description.resolve_attachments
                else None,
                vk.VkAttachmentReference(
                    attachment=subpass_description.depth_attachment_index,
                    layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                )
                if subpass_description.depth_attachment_index is not None
                else None,
            )
            for subpass_description in subpass_descriptions
        ]
        subpasses = [
            vk.VkSubpassDescription(
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                pColorAttachments=color_attachments,
                pResolveAttachments=resolve_attachments,
                pDepthStencilAttachment=depth_attachment,
            )
            for subpass_description, (
                color_attachments,
                resolve_attachments,
                depth_attachment,
            ) in zip(subpass_descriptions, native_resources)
        ]
        return kt.RenderPass(
            vk.vkCreateRenderPass(
                self.context.device,
                vk.VkRenderPassCreateInfo(
                    pAttachments=attachments, pSubpasses=subpasses
                ),
                None,
            )
        )

    @_add_to_resources
    def new_sampler(
        self, *, min_filter: kt.Filter, mag_filter: kt.Filter
    ) -> kt.Sampler:
        return kt.Sampler(
            vk.vkCreateSampler(
                self.context.device,
                vk.VkSamplerCreateInfo(
                    minFilter=min_filter,
                    magFilter=mag_filter,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    maxLod=float("inf"),
                ),
                None,
            )
        )

    def new_shader_set(self, *paths: str) -> typing.Tuple:
        filenames = [path.split("/")[-1] for path in paths]
        stages = [filename.split(".")[-2] for filename in filenames]
        attribute_names = ["_".join(filename.split(".")[:2]) for filename in filenames]
        spirvs = [
            shaderc.compile_shader(path=path, stage=stage)
            for path, stage in zip(paths, stages)
        ]
        shader_modules = [
            vk.vkCreateShaderModule(
                self.context.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv),
                None,
            )
            for spirv in spirvs
        ]
        self.context.resources.extend(shader_modules)
        return collections.namedtuple("ShaderSet", attribute_names)(*shader_modules)

    def update_descriptor_sets(
        self,
        buffer_writes: typing.Optional[typing.List[kt.DescriptorBufferWrites]] = None,
        image_writes: typing.Optional[typing.List[kt.DescriptorImageWrites]] = None,
    ) -> None:
        descriptor_writes = []

        if buffer_writes:
            buffer_write_infos = [
                [
                    vk.VkDescriptorBufferInfo(
                        buffer=buffer_info.buffer,
                        offset=buffer_info.byte_offset,
                        range=buffer_info.byte_count,
                    )
                    for buffer_info in buffer_write.buffer_infos
                ]
                for buffer_write in buffer_writes
            ]

            descriptor_writes.extend(
                [
                    vk.VkWriteDescriptorSet(
                        dstSet=buffer_write.descriptor_set,
                        dstBinding=buffer_write.binding,
                        descriptorCount=buffer_write.count,
                        descriptorType=buffer_write.descriptor_type,
                        pBufferInfo=buffer_infos,
                    )
                    for buffer_write, buffer_infos in zip(
                        buffer_writes, buffer_write_infos
                    )
                ]
            )

        if image_writes:
            image_write_infos = [
                [
                    vk.VkDescriptorImageInfo(
                        imageView=image_info.image_view, imageLayout=image_info.layout
                    )
                    for image_info in image_write.image_infos
                ]
                for image_write in image_writes
            ]

            descriptor_writes.extend(
                [
                    vk.VkWriteDescriptorSet(
                        dstSet=image_write.descriptor_set,
                        dstBinding=image_write.binding,
                        descriptorCount=image_write.count,
                        descriptorType=image_write.descriptor_type,
                        pImageInfo=image_infos,
                    )
                    for image_write, image_infos in zip(image_writes, image_write_infos)
                ]
            )

        vk.vkUpdateDescriptorSets(
            self.context.device,
            descriptorWriteCount=len(descriptor_writes),
            pDescriptorWrites=descriptor_writes,
            descriptorCopyCount=0,
            pDescriptorCopies=[],
        )


@contextlib.contextmanager
def run_graphics() -> typing.Generator[GraphicsApp, None, None]:
    errors: typing.List[str] = []

    def _debug_callback(
        _severity: int, _type: int, callback_data: typing.Any, _user_data: typing.Any
    ) -> int:
        message_string = vk.ffi.string(callback_data.pMessage).decode("utf-8")
        if not errors or errors[-1] != message_string:
            errors.append(message_string)
        return 0

    def create_memory_types(
        physical_device_memory_properties: typing.Any
    ) -> typing.Dict[kt.MemoryType, int]:
        memory_types = vk.ffi.unpack(
            physical_device_memory_properties.memoryTypes,
            physical_device_memory_properties.memoryTypeCount,
        )

        def get_memory_type_bits(memory_type_bit: int) -> int:
            return sum(
                bool(memory_type.propertyFlags & memory_type_bit) << index
                for index, memory_type in enumerate(memory_types)
            )

        def try_mask(flags: int, mask: int) -> int:
            return flags & mask if flags & mask else flags

        device_local = get_memory_type_bits(vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        host_visible = get_memory_type_bits(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        host_coherent = get_memory_type_bits(vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        host_cached = get_memory_type_bits(vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        lazily_allocated = get_memory_type_bits(
            vk.VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
        )

        device_optimal = try_mask(device_local, ~host_visible)
        uploadable = try_mask(host_visible, device_local)
        uploadable = try_mask(uploadable, host_coherent)
        uploadable = try_mask(uploadable, ~host_cached)
        downloadable = try_mask(host_visible, device_local)
        downloadable = try_mask(downloadable, host_cached)
        downloadable = try_mask(downloadable, ~host_coherent)
        lazily_allocated = lazily_allocated if lazily_allocated else device_local

        return {
            kt.MemoryType.DeviceOptimal: device_optimal,
            kt.MemoryType.Uploadable: uploadable,
            kt.MemoryType.Downloadable: downloadable,
            kt.MemoryType.LazilyAllocated: lazily_allocated,
        }

    debug_utils_messenger_create_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
        messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
        | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        pfnUserCallback=_debug_callback,
    )
    instance = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(
            pNext=debug_utils_messenger_create_info,
            pApplicationInfo=vk.VkApplicationInfo(
                pApplicationName="",
                applicationVersion=vk.VK_MAKE_VERSION(420, 0, 0),
                pEngineName="weed",
                engineVersion=vk.VK_MAKE_VERSION(69, 0, 0),
                apiVersion=vk.VK_MAKE_VERSION(1, 1, 110),
            ),
            ppEnabledExtensionNames=INSTANCE_EXTENSIONS,
            ppEnabledLayerNames=INSTANCE_LAYERS,
        ),
        None,
    )

    debug_utils_messenger = vk.vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT"
    )(instance, debug_utils_messenger_create_info, None)

    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    physical_devices_properties = [
        vk.vkGetPhysicalDeviceProperties(physical_device)
        for physical_device in physical_devices
    ]

    selected_physical_device_index = next(
        (
            index
            for index, physical_device_properties in enumerate(
                physical_devices_properties
            )
            if physical_device_properties.deviceType
            == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        ),
        0,
    )
    physical_device = physical_devices[selected_physical_device_index]

    graphics_queue_family_index = [
        index
        for index, properties in enumerate(
            vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
        )
        if properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
    ][0]

    device = vk.vkCreateDevice(
        physical_device,
        vk.VkDeviceCreateInfo(
            pQueueCreateInfos=[
                vk.VkDeviceQueueCreateInfo(
                    queueFamilyIndex=graphics_queue_family_index, pQueuePriorities=[1.0]
                )
            ],
            ppEnabledExtensionNames=DEVICE_EXTENSIONS,
            ppEnabledLayerNames=INSTANCE_LAYERS,
        ),
        None,
    )

    vk_bind_buffer_memory2 = vk.vkGetDeviceProcAddr(device, "vkBindBufferMemory2KHR")
    vk_bind_image_memory2 = vk.vkGetDeviceProcAddr(device, "vkBindImageMemory2KHR")

    graphics_queue = Queue(vk.vkGetDeviceQueue(device, graphics_queue_family_index, 0))

    memory_types = create_memory_types(
        vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    )

    allocations: typing.Dict[object, kt.DeviceMemory] = {}
    resources: typing.List[VulkanResource] = []

    try:
        yield GraphicsApp(
            VulkanContext(
                instance=instance,
                physical_device=physical_device,
                graphics_queue_family_index=graphics_queue_family_index,
                graphics_queue=graphics_queue,
                device=device,
                vk_bind_buffer_memory2=vk_bind_buffer_memory2,
                vk_bind_image_memory2=vk_bind_image_memory2,
                memory_types=memory_types,
                errors=errors,
                allocations=allocations,
                resources=resources,
            )
        )
    finally:
        destructors: typing.Dict[
            VulkanResource, typing.Callable[[kt.Device, VulkanResource, object], None]
        ] = {}
        for resource in resources:
            resource_type = vk.ffi.typeof(resource)
            if resource_type not in destructors:
                type_basename = resource_type.cname[9:-4]
                destructors[resource_type] = getattr(vk, "vkDestroy" + type_basename)
            destructors[resource_type](device, resource, None)

        for memory in (
            memory
            for allocations_list in allocations.values()
            for memory in allocations_list
        ):
            vk.vkFreeMemory(device, memory, None)

        vk.vkDestroyDevice(device, None)
        vk.vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")(
            instance, debug_utils_messenger, None
        )
        vk.vkDestroyInstance(instance, None)
