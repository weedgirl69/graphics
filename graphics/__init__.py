import collections
from enum import Enum
from typing import Dict, List, Tuple
from vulkan import ffi
import vulkan as vk
import pyshaderc
import graphics.types

INSTANCE_EXTENSIONS = ["VK_EXT_debug_utils"]
INSTANCE_LAYERS = ["VK_LAYER_LUNARG_standard_validation"]
DEVICE_EXTENSIONS = ["VK_KHR_bind_memory2"]


BUFFER_TYPE = ffi.typeof("struct VkBuffer_T *")
IMAGE_TYPE = ffi.typeof("struct VkImage_T *")


class MemoryType(Enum):
    DeviceOptimal = 0
    Uploadable = 1
    Downloadable = 2
    LazilyAllocated = 3


class CommandBufferBuilder:
    def __init__(self, command_buffer, flags):
        self.command_buffer = command_buffer
        self.flags = flags

    def __enter__(self):
        vk.vkBeginCommandBuffer(
            self.command_buffer, vk.VkCommandBufferBeginInfo(flags=self.flags)
        )

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        vk.vkEndCommandBuffer(self.command_buffer)

    def bind_descriptor_sets(self, pipeline_layout, descriptor_sets):
        vk.vkCmdBindDescriptorSets(
            self.command_buffer,
            vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            len(descriptor_sets),
            descriptor_sets,
            0,
            None,
        )

    def begin_render_pass(
        self,
        *,
        render_pass=None,
        framebuffer=None,
        width: int = None,
        height: int = None,
        clear_values=List[vk.VkClearValue],
    ):
        vk.vkCmdBeginRenderPass(
            self.command_buffer,
            vk.VkRenderPassBeginInfo(
                renderPass=render_pass,
                framebuffer=framebuffer,
                renderArea=vk.VkRect2D(extent=vk.VkExtent2D(width, height)),
                pClearValues=clear_values,
            ),
            vk.VK_SUBPASS_CONTENTS_INLINE,
        )

    def bind_pipeline(self, pipeline):
        vk.vkCmdBindPipeline(
            self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline
        )

    def bind_index_buffer(self, *, buffer, index_type, byte_offset: int = 0):
        vk.vkCmdBindIndexBuffer(self.command_buffer, buffer, byte_offset, index_type)

    def bind_vertex_buffers(
        self, buffers: List[object], byte_offsets: List[int] = None
    ):
        if not byte_offsets:
            byte_offsets = [0] * len(buffers)
        vk.vkCmdBindVertexBuffers(
            self.command_buffer, 0, len(buffers), buffers, byte_offsets
        )

    def blit_image(
        self,
        *,
        source_image,
        source_subresource_index,
        source_width,
        source_height,
        destination_image,
        destination_subresource_index,
        destination_width,
        destination_height,
    ):
        vk.vkCmdBlitImage(
            self.command_buffer,
            source_image,
            graphics.types.ImageLayout.TRANSFER_SOURCE,
            destination_image,
            graphics.types.ImageLayout.TRANSFER_DESTINATION,
            1,
            [
                vk.VkImageBlit(
                    srcSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=graphics.types.ImageAspect.COLOR,
                        mipLevel=source_subresource_index,
                        layerCount=1,
                    ),
                    srcOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(source_width, source_height, 1),
                    ],
                    dstSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=graphics.types.ImageAspect.COLOR,
                        mipLevel=destination_subresource_index,
                        layerCount=1,
                    ),
                    dstOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(destination_width, destination_height, 1),
                    ],
                )
            ],
            graphics.types.Filter.LINEAR,
        )

    def clear_color_image(
        self, *, image=None, color: Tuple[float, float, float, float] = (0, 0, 0, 0)
    ):
        vk.vkCmdClearColorImage(
            self.command_buffer,
            image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            vk.VkClearColorValue(color),
            1,
            vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1
            ),
        )

    def copy_buffer_to_image(self, *, buffer, image, width, height):
        vk.vkCmdCopyBufferToImage(
            self.command_buffer,
            buffer,
            image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            [
                vk.VkBufferImageCopy(
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
                    ),
                    imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
                )
            ],
        )

    def copy_image_to_buffer(
        self, *, image=None, buffer=None, width: int = 0, height: int = 0
    ):
        vk.vkCmdCopyImageToBuffer(
            self.command_buffer,
            image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            buffer,
            1,
            [
                vk.VkBufferImageCopy(
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
                    ),
                    imageExtent=vk.VkExtent3D(width, height, 1),
                )
            ],
        )

    def draw(
        self,
        *,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ):
        vk.vkCmdDraw(
            self.command_buffer,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        )

    def draw_indexed(
        self,
        *,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        vertex_offset: int = 0,
        first_instance: int = 0,
    ):
        vk.vkCmdDrawIndexed(
            self.command_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )

    def end_render_pass(self):
        vk.vkCmdEndRenderPass(self.command_buffer)

    def pipeline_barrier(
        self,
        *,
        image,
        old_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        new_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        base_mip_level: int = 0,
        mip_count: int = 1,
    ):
        vk.vkCmdPipelineBarrier(
            self.command_buffer,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [
                vk.VkImageMemoryBarrier(
                    srcAccessMask=0,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=old_layout,
                    newLayout=new_layout,
                    image=image,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=base_mip_level,
                        levelCount=mip_count,
                        layerCount=1,
                    ),
                )
            ],
        )


class Queue:
    def __init__(self, queue):
        self.queue = queue

    def submit(self, command_buffer):
        vk.vkQueueSubmit(
            self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[command_buffer])], None
        )

    def wait(self):
        vk.vkQueueWaitIdle(self.queue)


class App:
    # pylint: disable=too-many-instance-attributes

    class _Decorators:
        # pylint: disable=too-few-public-methods
        @staticmethod
        def add_to_resources(decorated):
            def wrapper(self, *args, **kwargs):
                resource = decorated(self, *args, **kwargs)
                self.resources.append(resource)
                return resource

            return wrapper

    def __init__(self):
        self.errors = []
        self.allocations = []
        self.resources = []

        def _debug_callback(_severity, _type, callback_data, _user_data):
            message_string = ffi.string(callback_data.pMessage).decode("utf-8")
            self.errors.append(message_string)
            return 0

        debug_utils_messenger_create_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
            messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            pfnUserCallback=_debug_callback,
            pUserData=ffi.new_handle(self),
        )

        self.instance = vk.vkCreateInstance(
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

        self.debug_utils_messenger = vk.vkGetInstanceProcAddr(
            self.instance, "vkCreateDebugUtilsMessengerEXT"
        )(self.instance, debug_utils_messenger_create_info, None)

        self.vk_destroy_debug_utils_messenger = vk.vkGetInstanceProcAddr(
            self.instance, "vkDestroyDebugUtilsMessengerEXT"
        )

        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
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
        self.selected_physical_device = physical_devices[selected_physical_device_index]

        self.graphics_queue_family_index = next(
            (
                index
                for index, properties in enumerate(
                    vk.vkGetPhysicalDeviceQueueFamilyProperties(
                        self.selected_physical_device
                    )
                )
                if properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
            )
        )

        self.device = vk.vkCreateDevice(
            self.selected_physical_device,
            vk.VkDeviceCreateInfo(
                pQueueCreateInfos=[
                    vk.VkDeviceQueueCreateInfo(
                        queueFamilyIndex=self.graphics_queue_family_index,
                        pQueuePriorities=[1.0],
                    )
                ],
                ppEnabledExtensionNames=DEVICE_EXTENSIONS,
                ppEnabledLayerNames=INSTANCE_LAYERS,
            ),
            None,
        )

        self.vk_bind_buffer_memory2 = vk.vkGetDeviceProcAddr(
            self.device, "vkBindBufferMemory2KHR"
        )
        self.vk_bind_image_memory2 = vk.vkGetDeviceProcAddr(
            self.device, "vkBindImageMemory2KHR"
        )

        self.graphics_queue = Queue(
            vk.vkGetDeviceQueue(self.device, self.graphics_queue_family_index, 0)
        )

        self.memory_types = App._create_memory_types(
            vk.vkGetPhysicalDeviceMemoryProperties(self.selected_physical_device)
        )

    def close(self):
        destructors = {}
        for resource in self.resources:
            resource_type = ffi.typeof(resource)
            if resource_type not in destructors:
                type_basename = resource_type.cname[9:-4]
                destructors[resource_type] = getattr(vk, "vkDestroy" + type_basename)
            destructors[resource_type](self.device, resource, None)

        for memory in self.allocations:
            vk.vkFreeMemory(self.device, memory, None)

        vk.vkDestroyDevice(self.device, None)

        self.vk_destroy_debug_utils_messenger(
            self.instance, self.debug_utils_messenger, None
        )

        vk.vkDestroyInstance(self.instance, None)

    def allocate_command_buffer(self, command_pool):
        command_buffers = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=command_pool, commandBufferCount=1
            ),
        )
        return command_buffers[0]

    def allocate_descriptor_set(self, *, descriptor_pool, descriptor_set_layouts=[]):
        return vk.vkAllocateDescriptorSets(
            self.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=descriptor_set_layouts,
            ),
        )

    @_Decorators.add_to_resources
    def new_buffer(self, *, byte_count, usage):
        return vk.vkCreateBuffer(
            self.device, vk.VkBufferCreateInfo(size=byte_count, usage=usage), None
        )

    @_Decorators.add_to_resources
    def new_command_pool(self):
        return vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                flags=vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                queueFamilyIndex=self.graphics_queue_family_index,
            ),
            None,
        )

    @_Decorators.add_to_resources
    def new_descriptor_pool(self):
        return vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1
                    )
                ],
            ),
            None,
        )

    @_Decorators.add_to_resources
    def new_descriptor_set_layout(self, immutable_samplers=None):
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                pImmutableSamplers=immutable_samplers,
            )
        ]
        return vk.vkCreateDescriptorSetLayout(
            self.device, vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings), None
        )

    @_Decorators.add_to_resources
    def new_framebuffer(
        self,
        *,
        render_pass,
        attachments: List[object],
        width: int,
        height: int,
        layers: int = 1,
    ):
        return vk.vkCreateFramebuffer(
            self.device,
            vk.VkFramebufferCreateInfo(
                renderPass=render_pass,
                pAttachments=attachments,
                width=width,
                height=height,
                layers=layers,
            ),
            None,
        )

    @_Decorators.add_to_resources
    def new_image(self, *, format, usage, width: int, height: int, mip_count: int = 1):
        return vk.vkCreateImage(
            self.device,
            vk.VkImageCreateInfo(
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=format,
                extent=vk.VkExtent3D(width, height, 1),
                mipLevels=mip_count,
                arrayLayers=1,
                usage=usage,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
            ),
            None,
        )

    @_Decorators.add_to_resources
    def new_image_view(
        self,
        *,
        image,
        format,
        mip_count: int = 1,
        aspect=graphics.types.ImageAspect.COLOR,
    ):
        return vk.vkCreateImageView(
            self.device,
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

    def new_memory_set(
        self,
        memory_requests: Dict[object, MemoryType],
        *,
        initial_values: Dict[object, bytes] = {},
    ):
        def select_memory_type_index(
            memory_requirements: vk.VkMemoryRequirements, memory_type: MemoryType
        ):
            allowable_memory_type_indices = (
                memory_requirements.memoryTypeBits & self.memory_types[memory_type]
            )
            assert allowable_memory_type_indices

            bits = bin(allowable_memory_type_indices)
            return len(bits) - len(bits.rstrip("0"))

        get_memory_requirements = {
            BUFFER_TYPE: vk.vkGetBufferMemoryRequirements,
            IMAGE_TYPE: vk.vkGetImageMemoryRequirements,
        }

        ResourceEntry = Tuple[object, vk.VkMemoryRequirements, int, int]
        buffers: List[ResourceEntry] = []
        images: List[ResourceEntry] = []
        type_to_resources: Dict[object, List[ResourceEntry]] = {
            BUFFER_TYPE: buffers,
            IMAGE_TYPE: images,
        }

        memory_type_index_to_byte_count: Dict[int, int] = collections.defaultdict(int)

        for resource, memory_type in memory_requests.items():
            resource_type = ffi.typeof(resource)

            memory_requirements = get_memory_requirements[resource_type](
                self.device, resource
            )
            memory_type_index = select_memory_type_index(
                memory_requirements, memory_type
            )

            type_to_resources[resource_type].append(
                (
                    resource,
                    memory_requirements,
                    memory_type_index,
                    memory_type_index_to_byte_count[memory_type_index],
                )
            )

            memory_type_index_to_byte_count[
                memory_type_index
            ] += memory_requirements.size

        memory_type_index_to_memory = {
            memory_type_index: vk.vkAllocateMemory(
                self.device,
                vk.VkMemoryAllocateInfo(
                    allocationSize=byte_count, memoryTypeIndex=memory_type_index
                ),
                None,
            )
            for memory_type_index, byte_count in memory_type_index_to_byte_count.items()
        }

        self.allocations += memory_type_index_to_memory.values()

        mappable_memory_types = (
            self.memory_types[MemoryType.Downloadable]
            | self.memory_types[MemoryType.Uploadable]
        )

        memory_type_index_to_mapped_memory = {
            memory_type_index: ffi.from_buffer(
                vk.vkMapMemory(
                    self.device,
                    memory_type_index_to_memory[memory_type_index],
                    0,
                    byte_count,
                    0,
                )
            )
            for memory_type_index, byte_count in memory_type_index_to_byte_count.items()
            if (1 << memory_type_index) & mappable_memory_types
        }

        if buffers:
            self.vk_bind_buffer_memory2(
                self.device,
                len(buffers),
                [
                    vk.VkBindBufferMemoryInfo(
                        buffer=buffer,
                        memory=memory_type_index_to_memory[memory_type_index],
                        memoryOffset=byte_offset,
                    )
                    for buffer, _, memory_type_index, byte_offset in buffers
                ],
            )

        if images:
            self.vk_bind_image_memory2(
                self.device,
                len(images),
                [
                    vk.VkBindImageMemoryInfo(
                        image=image,
                        memory=memory_type_index_to_memory[memory_type_index],
                        memoryOffset=byte_offset,
                    )
                    for image, _, memory_type_index, byte_offset in images
                ],
            )

        mappings = {
            resource: ffi.buffer(
                memory_type_index_to_mapped_memory[memory_type_index] + byte_offset,
                memory_requirements.size,
            )
            for resources in type_to_resources.values()
            for resource, memory_requirements, memory_type_index, byte_offset in resources
            if memory_type_index in memory_type_index_to_mapped_memory
        }

        for resource, value in initial_values.items():
            mappings[resource][: len(value)] = value

        return mappings

    @_Decorators.add_to_resources
    def new_pipeline(self, pipeline_description):
        return vk.vkCreateGraphicsPipelines(
            self.device, None, 1, [pipeline_description[0]], None
        )

    @_Decorators.add_to_resources
    def new_pipeline_layout(self, descriptor_set_layouts: List[object] = None):
        return vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(pSetLayouts=descriptor_set_layouts),
            None,
        )

    @_Decorators.add_to_resources
    def new_render_pass(self, *, attachments, subpass_descriptions):
        return vk.vkCreateRenderPass(
            self.device,
            vk.VkRenderPassCreateInfo(
                pAttachments=attachments, pSubpasses=subpass_descriptions
            ),
            None,
        )

    @_Decorators.add_to_resources
    def new_sampler(self, *, min_filter, mag_filter):
        return vk.vkCreateSampler(
            self.device,
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

    def new_shader_set(self, *paths):
        filenames = [path.split("/")[-1] for path in paths]
        stages = [filename.split(".")[-2] for filename in filenames]
        attribute_names = ["_".join(filename.split(".")[:2]) for filename in filenames]
        spirvs = [
            pyshaderc.compile_file_into_spirv(
                filepath=path, stage=stage, warnings_as_errors=True
            )
            for path, stage in zip(paths, stages)
        ]
        shader_modules = [
            vk.vkCreateShaderModule(
                self.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv),
                None,
            )
            for spirv in spirvs
        ]
        self.resources += shader_modules
        return collections.namedtuple("ShaderSet", attribute_names)(*shader_modules)

    def update_descriptor_sets(self, descriptor_set_writes: List[object]):
        vk.vkUpdateDescriptorSets(
            self.device,
            descriptorWriteCount=len(descriptor_set_writes),
            pDescriptorWrites=descriptor_set_writes,
            descriptorCopyCount=0,
            pDescriptorCopies=[],
        )

    @staticmethod
    def _create_memory_types(physical_device_memory_properties):
        memory_types = ffi.unpack(
            physical_device_memory_properties.memoryTypes,
            physical_device_memory_properties.memoryTypeCount,
        )

        def get_memory_type_bits(memory_type_bit):
            return sum(
                bool(memory_type.propertyFlags & memory_type_bit) << index
                for index, memory_type in enumerate(memory_types)
            )

        def try_mask(flags, mask):
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
            MemoryType.DeviceOptimal: device_optimal,
            MemoryType.Uploadable: uploadable,
            MemoryType.Downloadable: downloadable,
            MemoryType.LazilyAllocated: lazily_allocated,
        }

