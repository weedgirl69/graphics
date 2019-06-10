from enum import Enum, auto
from vulkan import *
import collections


INSTANCE_EXTENSIONS = ["VK_EXT_debug_utils"]
INSTANCE_LAYERS = ["VK_LAYER_LUNARG_standard_validation"]
BUFFER_TYPE = ffi.typeof("struct VkBuffer_T *")
IMAGE_TYPE = ffi.typeof("struct VkImage_T *")


class MemoryType(Enum):
    DeviceOptimal = auto()
    Uploadable = auto()
    Downloadable = auto()
    LazilyAllocated = auto()


class ImageLayout:
    TRANSFER_SOURCE = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    TRANSFER_DESTINATION = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL


class MemoryBuilder:
    def __init__(self, app):
        self.app = app
        self.memory_type_index_to_byte_count = collections.defaultdict(int)
        self.buffers = []
        self.images = []
        self.resources = {BUFFER_TYPE: self.buffers, IMAGE_TYPE: self.images}
        self.resource_to_memory_type_index_and_byte_offset = {}

    def add(self, resource, memory_type: MemoryType):
        get_memory_requirements = {
            BUFFER_TYPE: vkGetBufferMemoryRequirements,
            IMAGE_TYPE: vkGetImageMemoryRequirements,
        }

        resource_type = ffi.typeof(resource)

        memory_requirements = get_memory_requirements[resource_type](
            self.app.device, resource
        )
        memory_type_index = self._select_memory_type_index(
            memory_requirements, memory_type
        )

        self.resources[resource_type].append(
            (
                resource,
                memory_requirements,
                memory_type_index,
                self.memory_type_index_to_byte_count[memory_type_index],
            )
        )

        self.memory_type_index_to_byte_count[
            memory_type_index
        ] += memory_requirements.size

        return self

    def build(self):
        memory_type_index_to_memory = {
            memory_type_index: self.app.allocate_memory(
                memory_type_index=memory_type_index, byte_count=byte_count
            )
            for memory_type_index, byte_count in self.memory_type_index_to_byte_count.items()
        }

        memory_type_index_to_mapped_memory = {
            memory_type_index: ffi.from_buffer(
                vkMapMemory(
                    self.app.device,
                    memory_type_index_to_memory[memory_type_index],
                    0,
                    byte_count,
                    0,
                )
            )
            for memory_type_index, byte_count in self.memory_type_index_to_byte_count.items()
            if (1 << memory_type_index) & self.app.memory_types[MemoryType.Downloadable]
        }

        for buffer, _, memory_type_index, byte_offset in self.buffers:
            vkBindBufferMemory(
                self.app.device,
                buffer,
                memory_type_index_to_memory[memory_type_index],
                byte_offset,
            )

        for image, _, memory_type_index, byte_offset in self.images:
            vkBindImageMemory(
                self.app.device,
                image,
                memory_type_index_to_memory[memory_type_index],
                byte_offset,
            )

        return {
            resource: ffi.buffer(
                memory_type_index_to_mapped_memory[memory_type_index] + byte_offset,
                memory_requirements.size,
            )
            for resources in self.resources.values()
            for resource, memory_requirements, memory_type_index, byte_offset in resources
            if memory_type_index in memory_type_index_to_mapped_memory
        }

    def _select_memory_type_index(
        self, memory_requirements: VkMemoryRequirements, memory_type: MemoryType
    ):
        allowable_memory_type_indices = (
            self.app.memory_types[memory_type] & memory_requirements.memoryTypeBits
        )
        assert allowable_memory_type_indices

        result = 0
        while not allowable_memory_type_indices & 1:
            allowable_memory_type_indices = allowable_memory_type_indices >> 1
            result += 1
        return result


class CommandBufferBuilder:
    ONE_TIME_SUBMIT = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT

    def __init__(self, command_buffer, flags):
        self.command_buffer = command_buffer
        self.flags = flags

    def __enter__(self):
        vkBeginCommandBuffer(
            self.command_buffer, VkCommandBufferBeginInfo(flags=self.flags)
        )

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        vkEndCommandBuffer(self.command_buffer)

    def clear_color_image(self, image, color):
        vkCmdClearColorImage(
            self.command_buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VkClearColorValue(color),
            1,
            VkImageSubresourceRange(
                aspectMask=VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1
            ),
        )

    def copy_image_to_buffer(self, image, buffer, width=0, height=0):
        vkCmdCopyImageToBuffer(
            self.command_buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            buffer,
            1,
            [
                VkBufferImageCopy(
                    imageSubresource=VkImageSubresourceLayers(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
                    ),
                    imageExtent=VkExtent3D(width, height, 1),
                )
            ],
        )

    def pipeline_barrier(
        self,
        image,
        old_layout=VK_IMAGE_LAYOUT_UNDEFINED,
        new_layout=VK_IMAGE_LAYOUT_UNDEFINED,
    ):
        vkCmdPipelineBarrier(
            self.command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [
                VkImageMemoryBarrier(
                    srcAccessMask=0,
                    dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=old_layout,
                    newLayout=new_layout,
                    image=image,
                    subresourceRange=VkImageSubresourceRange(
                        aspectMask=VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1
                    ),
                )
            ],
        )


class CommandBuffer:
    def __init__(self, command_buffer):
        self.command_buffer = command_buffer

    def build(self, flags):
        return CommandBufferBuilder(self.command_buffer, flags)


class Queue:
    def __init__(self, queue):
        self.queue = queue

    def submit(self, command_buffer):
        vkQueueSubmit(
            self.queue,
            1,
            [VkSubmitInfo(pCommandBuffers=[command_buffer.command_buffer])],
            None,
        )

    def wait(self):
        vkQueueWaitIdle(self.queue)


class App:
    def __init__(self):
        self.has_errors = False

        def _debug_callback(severity, type, callback_data, user_data):
            print(ffi.string(callback_data.pMessage).decode("utf-8"))
            self.has_errors = True
            return 0

        debug_utils_messenger_create_info = VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
            messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            pfnUserCallback=_debug_callback,
            pUserData=ffi.new_handle(self),
        )

        self.instance = vkCreateInstance(
            VkInstanceCreateInfo(
                pNext=debug_utils_messenger_create_info,
                pApplicationInfo=VkApplicationInfo(
                    pApplicationName="",
                    applicationVersion=VK_MAKE_VERSION(420, 0, 0),
                    pEngineName="weed",
                    engineVersion=VK_MAKE_VERSION(69, 0, 0),
                    apiVersion=VK_MAKE_VERSION(1, 1, 0),
                ),
                ppEnabledExtensionNames=INSTANCE_EXTENSIONS,
                ppEnabledLayerNames=INSTANCE_LAYERS,
            ),
            None,
        )

        vkCreateDebugUtilsMessengerEXT = vkGetInstanceProcAddr(
            self.instance, "vkCreateDebugUtilsMessengerEXT"
        )
        self.vkDestroyDebugUtilsMessengerEXT = vkGetInstanceProcAddr(
            self.instance, "vkDestroyDebugUtilsMessengerEXT"
        )

        self.debug_utils_messenger = vkCreateDebugUtilsMessengerEXT(
            self.instance, debug_utils_messenger_create_info, None
        )

        physical_devices = vkEnumeratePhysicalDevices(self.instance)
        physical_devices_properties = [
            vkGetPhysicalDeviceProperties(physical_device)
            for physical_device in physical_devices
        ]

        selected_physical_device_index = next(
            (
                index
                for index, (physical_device, properties) in enumerate(
                    zip(physical_devices, physical_devices_properties)
                )
                if properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            ),
            0,
        )
        self.selected_physical_device = physical_devices[selected_physical_device_index]
        self.selected_physical_device_properties = physical_devices_properties[
            selected_physical_device_index
        ]

        self.graphics_queue_family_index = next(
            (
                index
                for index, properties in enumerate(
                    vkGetPhysicalDeviceQueueFamilyProperties(
                        self.selected_physical_device
                    )
                )
                if properties.queueFlags & VK_QUEUE_GRAPHICS_BIT
            )
        )

        self.device = vkCreateDevice(
            self.selected_physical_device,
            VkDeviceCreateInfo(
                pQueueCreateInfos=[
                    VkDeviceQueueCreateInfo(
                        queueFamilyIndex=self.graphics_queue_family_index,
                        pQueuePriorities=[1],
                    )
                ],
                ppEnabledLayerNames=INSTANCE_LAYERS,
            ),
            None,
        )

        self.graphics_queue = Queue(
            vkGetDeviceQueue(self.device, self.graphics_queue_family_index, 0)
        )

        self.memory_types = App._create_memory_types(
            vkGetPhysicalDeviceMemoryProperties(self.selected_physical_device)
        )

        self.buffers = set()
        self.command_pools = set()
        self.images = set()
        self.memorys = set()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for buffer in self.buffers:
            vkDestroyBuffer(self.device, buffer, None)
        for command_pool in self.command_pools:
            vkDestroyCommandPool(self.device, command_pool, None)
        for image in self.images:
            vkDestroyImage(self.device, image, None)
        for memory in self.memorys:
            vkFreeMemory(self.device, memory, None)
        vkDestroyDevice(self.device, None)

        self.vkDestroyDebugUtilsMessengerEXT(
            self.instance, self.debug_utils_messenger, None
        )

        vkDestroyInstance(self.instance, None)

        assert not self.has_errors

    def new_render_target(self, width, height):
        image = vkCreateImage(
            self.device,
            VkImageCreateInfo(
                imageType=VK_IMAGE_TYPE_2D,
                format=VK_FORMAT_R8G8B8A8_UNORM,
                extent=VkExtent3D(width, height, 1),
                mipLevels=1,
                arrayLayers=1,
                usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                samples=VK_SAMPLE_COUNT_1_BIT,
            ),
            None,
        )
        self.images.add(image)
        return image

    def new_readback_buffer(self, byte_count):
        buffer = vkCreateBuffer(
            self.device,
            VkBufferCreateInfo(size=byte_count, usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            None,
        )
        self.buffers.add(buffer)
        return buffer

    def memory_builder(self):
        return MemoryBuilder(self)

    def allocate_memory(self, memory_type_index=None, byte_count=None):
        memory = vkAllocateMemory(
            self.device,
            VkMemoryAllocateInfo(
                allocationSize=byte_count, memoryTypeIndex=memory_type_index
            ),
            None,
        )

        self.memorys.add(memory)
        return memory

    def new_command_pool(self):
        command_pool = vkCreateCommandPool(
            self.device,
            VkCommandPoolCreateInfo(
                flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                queueFamilyIndex=self.graphics_queue_family_index,
            ),
            None,
        )
        self.command_pools.add(command_pool)
        return command_pool

    def allocate_command_buffer(self, command_pool):
        command_buffers = vkAllocateCommandBuffers(
            self.device,
            VkCommandBufferAllocateInfo(commandPool=command_pool, commandBufferCount=1),
        )
        return CommandBuffer(command_buffers[0])

    @staticmethod
    def _create_memory_types(physical_device_memory_properties):
        memory_types = ffi.unpack(
            physical_device_memory_properties.memoryTypes,
            physical_device_memory_properties.memoryTypeCount,
        )

        get_memory_type_bits = lambda memory_type_bit: sum(
            (
                bool(memory_type.propertyFlags & memory_type_bit) << index
                for index, memory_type in enumerate(memory_types)
            )
        )

        try_mask = lambda flags, mask: flags & mask if flags & mask else flags

        device_local = get_memory_type_bits(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        host_visible = get_memory_type_bits(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        host_coherent = get_memory_type_bits(VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        host_cached = get_memory_type_bits(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        lazily_allocated = get_memory_type_bits(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)

        device_optimal = try_mask(device_local, ~host_visible)
        uploadable = try_mask(host_visible, device_local)
        uploadable = try_mask(uploadable, host_coherent)
        uploadable = try_mask(uploadable, ~host_cached)
        downloadable = try_mask(host_visible, device_local)
        downloadable = try_mask(downloadable, host_cached)
        downloadable = try_mask(downloadable, ~host_coherent)

        return {
            MemoryType.DeviceOptimal: device_optimal,
            MemoryType.Uploadable: uploadable,
            MemoryType.Downloadable: downloadable,
            MemoryType.LazilyAllocated: lazily_allocated,
        }

