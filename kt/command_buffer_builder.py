from __future__ import annotations
import typing
import vulkan as vk
import kt


class CommandBufferBuilder:
    def __init__(self, *, command_buffer: kt.CommandBuffer, usage: int) -> None:
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
        self,
        pipeline_layout: kt.PipelineLayout,
        descriptor_sets: typing.List[kt.DescriptorSet],
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
        render_pass: kt.RenderPass,
        framebuffer: kt.Framebuffer,
        width: int,
        height: int,
        clear_values: typing.List[typing.Union[kt.ClearColor, kt.ClearDepth]],
    ) -> None:
        clear_values = [
            {
                kt.ClearColor: lambda x: vk.VkClearValue(
                    color=vk.VkClearColorValue(
                        float32=(x.red, x.green, x.blue, x.alpha)
                    )
                ),
                kt.ClearDepth: lambda x: vk.VkClearValue(
                    depthStencil=vk.VkClearDepthStencilValue(depth=x.depth)
                ),
            }[type(clear_value)](clear_value)
            for clear_value in clear_values
        ]
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

    def bind_pipeline(self, pipeline: kt.Pipeline) -> None:
        vk.vkCmdBindPipeline(
            self.command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline
        )

    def bind_index_buffer(
        self, *, buffer: kt.Buffer, index_type: kt.IndexType, byte_offset: int = 0
    ) -> None:
        vk.vkCmdBindIndexBuffer(self.command_buffer, buffer, byte_offset, index_type)

    def bind_vertex_buffers(
        self, buffers: typing.List[kt.Buffer], byte_offsets: typing.List[int] = None
    ) -> None:
        if not byte_offsets:
            byte_offsets = [0] * len(buffers)
        vk.vkCmdBindVertexBuffers(
            self.command_buffer, 0, len(buffers), buffers, byte_offsets
        )

    def blit_image(
        self,
        *,
        source_image: kt.Image,
        source_subresource_index: int = 0,
        source_width: int,
        source_height: int,
        destination_image: kt.Image,
        destination_subresource_index: int = 0,
        destination_width: int,
        destination_height: int,
    ) -> None:
        vk.vkCmdBlitImage(
            self.command_buffer,
            source_image,
            kt.ImageLayout.TRANSFER_SOURCE,
            destination_image,
            kt.ImageLayout.TRANSFER_DESTINATION,
            1,
            [
                vk.VkImageBlit(
                    srcSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=kt.ImageAspect.COLOR,
                        mipLevel=source_subresource_index,
                        layerCount=1,
                    ),
                    srcOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(source_width, source_height, 1),
                    ],
                    dstSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=kt.ImageAspect.COLOR,
                        mipLevel=destination_subresource_index,
                        layerCount=1,
                    ),
                    dstOffsets=[
                        vk.VkOffset3D(),
                        vk.VkOffset3D(destination_width, destination_height, 1),
                    ],
                )
            ],
            kt.Filter.LINEAR,
        )

    def clear_color_image(
        self,
        *,
        image: kt.Image,
        color: typing.Tuple[float, float, float, float] = (0, 0, 0, 0),
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

    def copy_buffer_to_buffer(
        self,
        *,
        source_buffer: kt.Buffer,
        source_offset: int = 0,
        destination_buffer: kt.Buffer,
        destination_offset: int = 0,
        byte_count: int,
    ) -> None:
        vk.vkCmdCopyBuffer(
            self.command_buffer,
            source_buffer,
            destination_buffer,
            regionCount=1,
            pRegions=[
                vk.VkBufferCopy(
                    srcOffset=source_offset,
                    dstOffset=destination_offset,
                    size=byte_count,
                )
            ],
        )

    def copy_buffer_to_image(
        self, *, buffer: kt.Buffer, image: kt.Image, width: int, height: int
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
        self, *, image: kt.Image, buffer: kt.Buffer, width: int, height: int
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
        image: kt.Image,
        old_layout: kt.ImageLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
        new_layout: kt.ImageLayout = vk.VK_IMAGE_LAYOUT_UNDEFINED,
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

    def push_constants(
        self,
        *,
        pipeline_layout: kt.PipelineLayout,
        stage: kt.ShaderStage,
        byte_offset: int,
        values: bytes,
    ) -> None:
        a = vk.ffi.from_buffer(values)
        vk.vkCmdPushConstants(
            self.command_buffer,
            layout=pipeline_layout,
            stageFlags=stage,
            offset=byte_offset,
            size=len(values),
            pValues=a,
        )
