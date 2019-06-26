from __future__ import annotations
from typing import List, Tuple
import vulkan as vk
from kt import (
    Buffer,
    ClearValue,
    CommandBuffer,
    DescriptorSet,
    Filter,
    Framebuffer,
    Image,
    ImageAspect,
    ImageLayout,
    Pipeline,
    PipelineLayout,
    IndexType,
    RenderPass,
)


class CommandBufferBuilder:
    def __init__(self, *, command_buffer: CommandBuffer, usage: int) -> None:
        self.command_buffer = command_buffer
        self.usage = usage

    def __enter__(self) -> CommandBufferBuilder:
        vk.vkBeginCommandBuffer(
            self.command_buffer, vk.VkCommandBufferBeginInfo(flags=self.usage)
        )

        return self

    def __exit__(
        self, exception_type: None, exception_value: None, traceback: None
    ) -> None:
        vk.vkEndCommandBuffer(self.command_buffer)

    def bind_descriptor_sets(
        self, pipeline_layout: PipelineLayout, descriptor_sets: List[DescriptorSet]
    ) -> None:
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
        render_pass: RenderPass,
        framebuffer: Framebuffer,
        width: int,
        height: int,
        clear_values: List[ClearValue],
    ) -> None:
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

    def bind_pipeline(self, pipeline: Pipeline) -> None:
        vk.vkCmdBindPipeline(
            self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline
        )

    def bind_index_buffer(
        self, *, buffer: Buffer, index_type: IndexType, byte_offset: int = 0
    ) -> None:
        vk.vkCmdBindIndexBuffer(self.command_buffer, buffer, byte_offset, index_type)

    def bind_vertex_buffers(
        self, buffers: List[Buffer], byte_offsets: List[int] = None
    ) -> None:
        if not byte_offsets:
            byte_offsets = [0] * len(buffers)
        vk.vkCmdBindVertexBuffers(
            self.command_buffer, 0, len(buffers), buffers, byte_offsets
        )

    def blit_image(
        self,
        *,
        source_image: Image,
        source_subresource_index: int = 0,
        source_width: int,
        source_height: int,
        destination_image: Image,
        destination_subresource_index: int = 0,
        destination_width: int,
        destination_height: int,
    ) -> None:
        vk.vkCmdBlitImage(
            self.command_buffer,
            source_image,
            ImageLayout.TRANSFER_SOURCE,
            destination_image,
            ImageLayout.TRANSFER_DESTINATION,
            1,
            [
                vk.VkImageBlit(
                    srcSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=ImageAspect.COLOR,
                        mipLevel=source_subresource_index,
                        layerCount=1,
                    ),
                    srcOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(source_width, source_height, 1),
                    ],
                    dstSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=ImageAspect.COLOR,
                        mipLevel=destination_subresource_index,
                        layerCount=1,
                    ),
                    dstOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(destination_width, destination_height, 1),
                    ],
                )
            ],
            Filter.LINEAR,
        )

    def clear_color_image(
        self, *, image: Image, color: Tuple[float, float, float, float] = (0, 0, 0, 0)
    ) -> None:
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

    def copy_buffer_to_image(
        self, *, buffer: Buffer, image: Image, width: int, height: int
    ) -> None:
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
        self, *, image: Image, buffer: Buffer, width: int, height: int
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        vk.vkCmdDrawIndexed(
            self.command_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        )

    def end_render_pass(self) -> None:
        vk.vkCmdEndRenderPass(self.command_buffer)

    def pipeline_barrier(
        self,
        *,
        image: Image,
        old_layout: ImageLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
        new_layout: ImageLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
        base_mip_level: int = 0,
        mip_count: int = 1,
    ) -> None:
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
