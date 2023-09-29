#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

/// Draws Viking rooms in a new window following the Vulkan graphics API tutorial at:
/// https://kylemayes.github.io/vulkanalia/introduction.html
///
/// The program starts with one model and pressing the right and left arrow keys displays more or
/// less models respectively. Models have increasing opacity.
///
/// To run with debugging output, use (on Linux):
/// RUST_LOG="debug" cargo run
///
/// Logging options starting with the most verbose are "trace", "debug", "info", "warn", "error".

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::{copy_nonoverlapping as memcpy, slice_from_raw_parts};
use std::time::Instant;

use anyhow::{anyhow, Result};
use cgmath::{Deg, point3, vec2, vec3};
use log::*;
use thiserror::Error;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::Version;
use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use vulkanalia::window as vk_window;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);  // True if cargo build enables debugging
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
const MAX_FRAMES_IN_FLIGHT: usize = 2;

type Mat4 = cgmath::Matrix4<f32>;
type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

/// Creates a new window with an associated event loop that polls for certain window events. The
/// events listened for are: window closure, window resizing and the user pressing either left or
/// right arrow keys.
fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new();  // EventLoop is implemented by winit

    // Create window.
    let window = WindowBuilder::new()
        .with_title("Vulkanalia Viking Room")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // Main event loop.
    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;  // True if user closes window or kills application
    let mut minimized = false;  // True if user minimizes window
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;  // Configure polling for the event loop
        match event {
            // Render a frame unless app is minimized or is in the process of being destroyed.
            Event::MainEventsCleared if !destroying && !minimized =>
                unsafe { app.render(&window) }.unwrap(),
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {  // Window resize
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {  // Destroy app
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.device.device_wait_idle().unwrap(); }
                unsafe { app.destroy(); }
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Left) if app.models > 1 => app.models -= 1,
                        Some(VirtualKeyCode::Right) if app.models < 4 => app.models += 1,
                        _ => { }
                    }
                }
            }
            _ => {}
        }
    });
}

/// The core application that sets up all the necessary resources, renders each frame and
/// destroys all resources.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,  // Current frame index
    resized: bool,  // True if window has been resized but swapchain has not yet been recreated
    start: Instant,  // Records the time the program started which is used to track elapsed time
    models: usize,  // Number of models to display
}

impl App {
    /// Creates the app. This consists of:
    ///     - Configuring the dynamic loading of the Vulkan shared library from the OS and using
    ///       this to create an entry point that allows Vulkan entry commands to be used.
    ///     - Creating a Vulkan instance that is configured to use the features and extensions
    ///       needed by the rest of the code. This is bound to the `window` passed to this
    ///       function.
    ///     - Creating a surface for the window. Presenting images to this surface displays them.
    ///     - Picking a physical graphics device to use. This requires the features of each device
    ///       to be examined to ensure all the features required for the rest of this program
    ///       are available. If the system contains multiple graphics devices, the most suitable is
    ///       chosen.
    ///     - Creating a logical device that specifies which optional features will be used by the
    ///       program.
    ///     - Creating a swapchain and an associated set of images that are used for image
    ///       rendering.
    ///     - Creating views to access the swapchain images.
    ///     - Creating a render pass that defines the attachments needed for rendering and the
    ///       order of operations.
    ///     - Creating a descriptor set layout that allows model, view and projection matrices to
    ///       be passed to shaders. The model matrix is passed via push constants, and the view and
    ///       projection matrices are passed in a single UBO (uniform buffer object).
    ///     - Creating a render pipeline that configures both fixed-function and programmable
    ///       pipeline stages.
    ///     - Creating command pools to hold command buffers that perform the operations such as
    ///       drawing the scene.
    ///     - Creating an image object with multi-sample anti-aliasing (MSAA) enabled to be used as
    ///       source data for rendering the final scene using MSAA.
    ///     - Creating a depth object to record depth information about the image as it is
    ///       rendered.
    ///     - Creating a set of framebuffers, one for each image in the swapchain.
    ///     - Creating a texture image to store a texture.
    ///     - Creating a texture image view to access the texture image.
    ///     - Creating a texture sampler to allow filters and effects to be applied as the shader
    ///       reads data from the texture image view.
    ///     - Loading a 3D model from the file system.
    ///     - Creating a buffer to hold vertex data that will be sent to the vertex shader.
    ///     - Creating a buffer to hold indices into the vertex data that will also be sent to the
    ///       vertex shader.
    ///     - Creating a uniform buffer to hold the view and projection matrices. (The model
    ///       matrix is passed using push constants).
    ///     - Creating a descriptor pool allowing the creation of a descriptor to the UBO.
    ///     - Creating command buffers that contain the actual drawing commands.
    ///     - Creating semaphore and fence objects to synchronize other code so that resources are
    ///       only used when not actively in use by asynchronous Vulkan operations.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;  // Load Vulkan library from OS
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;  // Vulkan entry point
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_command_pools(&instance, &device, &mut data)?;
        create_color_objects(&instance, &device, &mut data)?;
        create_depth_objects(&instance, &device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_texture_image_view(&device, &mut data)?;
        create_texture_sampler(&device, &mut data)?;
        load_model(&mut data)?;
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 1,
        })
    }

    /// Renders a frame.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // Waits until the fence for this frame is signaled, indicating that it has been displayed,
        // and therefore its semaphores are ready for reuse.
        self.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame]],  // The fence associated with this frame
            true,  // Wait for all fences? Doesn't matter as we only pass one fence in array
            u64::max_value(),  // Don't set a timeout
        )?;

        // Attempts to acquire an available swapchain image.
        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::max_value(),  // Don't set a timeout for waiting for image acquisition
            self.data.image_available_semaphores[self.frame],  // Indicates when image can be used
            vk::Fence::null(),
        );

        // If the attempt to acquire an available swapchain image is successful, stores its index.
        // If not, the swapchain is recreated if the return value indicates it is outdated.
        // Other errors are propagated.
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // A swapchain image is available for rendering, so checks if it currently has an
        // associated fence. If not, obtains an available fence, which may involve waiting for
        // one to become available if all are currently associated with in-flight frames. This acts
        // as a throttle on the number of frames that are in-flight at the same time.
        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::max_value(),
            )?;
        }

        // Overwrites the fence reference for this swapchain image to refer to the fence the frame
        // is now associated with. This is how we map images to frames.
        self.data.images_in_flight[image_index as usize] =
            self.data.in_flight_fences[self.frame];

        // Resets the command buffer for this image, erasing all previous data.
        self.update_command_buffer(image_index)?;

        // Updates the uniform buffers that hold the view and projection matrices.
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        // Resets the fence's state back to unsignaled.
        self.device.reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        // Submits the commands in the command queue to be run. The last argument is the fence
        // that is signaled when the commands complete.
        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.swapchain];  // Note: we only create one swapchain
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)  // The semaphores to wait on
            .swapchains(swapchains)  // The swapchain to apply images to
            .image_indices(image_indices);  // The index of the image to present

        // Attempts to queue an image to be presented, i.e., displayed.
        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);

        // If window changes mean that the current swapchain is suboptimal or outdated, it is
        // recreated and no rendering is performed. The next render cycle will use the new
        // swapchain and should succeed unless a further change is made.
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        // Increment the frame index ready for the next frame render.
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Destroys the resources managed by this `App`.
    //
    // In general, resources are destroyed in the reverse order they were created.
    unsafe fn destroy(&mut self) {
        self.destroy_swapchain();
        self.data.command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device.free_memory(self.data.vertex_buffer_memory, None);
        self.data.in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys resources associated with a swapchain, then the swapchain itself.
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_image_view(self.data.color_image_view, None);  // Used for MSAA
        self.device.free_memory(self.data.color_image_memory, None);  // Used for MSAA
        self.device.destroy_image(self.data.color_image, None);  // Used for MSAA
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));
        self.data.uniform_buffers_memory
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));
        self.data.framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    /// Recreates the swapchain. This is used when changes are made that cannot be reflected by
    /// modifying the swapchain, the most common being the user resizing the window.
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;

        // Resizes the array of images in flight in case the window change results in a different
        // number of swapchain images (which is unlikely).
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    /// Updates the view and projection matrices in the UBO (uniform buffer object)
    /// associated with the `image_index` passed. The model matrix is rotated at a rate
    /// dependent on the program's runtime.
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();  // Seconds since program started

        // Creates a model matrix that is a rotation around the positive Z-axis based on the time
        // elapsed since program start.
        let model = Mat4::from_axis_angle(
            vec3(0.0, 0.0, 1.0),  // Axis around which to perform rotation
            Deg(90.0) * time
        );

        // Creates a view matrix that positions the camera so that it is looking directly at the
        // YZ plane.
        let view = Mat4::look_at_rh(
            point3(6.0, 0.0, 2.0),  // Eye position
            point3(0.0, 0.0, 0.0),  // Look toward this position
            vec3(0.0, 0.0, 1.0),  // A vector defining "up"
        );

        // cgmath uses the OpenGL standard where the Y axis of the clip coordinates is
        // inverted. Additionally, the depth buffer range is -1.0..1.0, but Vulkan uses 0.0..1.0.
        // The following correction matrix fixes both problems, the former by flipping the Y-axis
        // to prevent the image being rendered upside down.
        // cgmath uses column-major ordering, meaning the translation on the last line is actually
        // the last value on the third row if using the more common row-major representation.
        #[rustfmt::skip]
        let correction = Mat4::new(
            1.0,  0.0,       0.0, 0.0,
            0.0, -1.0,       0.0, 0.0,  // Flips the Y-axis
            0.0,  0.0, 1.0 / 2.0, 0.0,
            0.0,  0.0, 1.0 / 2.0, 1.0,
        );

        let proj = correction * cgmath::perspective(
            Deg(45.0),  // 45 degree vertical field-of-view
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,  // Near edge clipping plane
            10.0,  // Far edge clipping plane
        );
        // The second parameter is the aspect ratio. This uses the current swapchain extent's
        // width and height in case the user resized the window from its initial size.

        // Creates a UBO with the view and projection matrices generated above.
        let ubo = UniformBufferObject { view, proj };

        // Maps the UBO memory to RAM so the CPU can copy data in.
        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        // Copies data into the UBO buffer.
        memcpy(&ubo, memory.cast(), 1);

        // Unmaps the UBO memory from RAM as the data is now in the former.
        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    /// Resets a command pool, which resets all command buffers it contains.
    /// This method must only be called when the command pool and all the command buffers it
    /// contains are all idle.
    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Reset the command pool associated with the image at the `image_index` passed.
        let command_pool = self.data.command_pools[image_index];
        self.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);  // May improve performance

        self.device.begin_command_buffer(command_buffer, &info)?;

        // Defines the size of the render area. In most cases, this should equal the size of
        // attachments.
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        // Sets black as the color to use when clearing an image.
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        // Sets the maximum depth value as the value to use when clearing the depth buffer.
        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        // Sets the clear values for the color and depth buffers. The order in the array must match
        // the order the attachments are specified.
        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        // Begins the rendering pass.
        self.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS, // Declares secondary command buff. use
        );

        // Creates secondary command buffers containing the commands necessary to render 4 versions
        // of the model, then executes them.
        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;
        self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        // Ends the rendering pass and finishes recording.
        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    /// Creates new secondary command buffers if the current count of them is less than the
    /// minimum needed, as passed in `model_index`.
    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        self.data.secondary_command_buffers.resize_with(image_index + 1, Vec::new);
        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)  // Secondary buffer
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // Creates inheritance information that is needed when creating secondary command buffers
        // (but not primary buffers). This defines which render pass, subpass index, and
        // framebuffer this secondary command buffer will be used for.
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);  // Optional. May improve perf.

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE) // See below
            .inheritance_info(&inheritance_info);
        // RENDER_PASS_CONTINUE specifies that this secondary command buffer will be executed
        // entirely inside a render pass.

        // Translate model so multiple models are not in exactly the same position.
        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let time = self.start.elapsed().as_secs_f32();  // Seconds since program started

        // Calculates the model projection so it can be passed to the vertex shader as a push
        // constant by setting the data with a command.
        let model = Mat4::from_translation(vec3(0.0, y, z)) * Mat4::from_axis_angle(
            vec3(0.0, 0.0, 1.0),
            Deg(90.0) * time
        );

        // Converts the matrix into bytes as the push constant is defined as a contiguous range of
        // 64 bytes.
        let model_bytes = &*slice_from_raw_parts(
            &model as *const Mat4 as *const u8,
            size_of::<Mat4>()
        );

        // Set different opacities for each model.
        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline
        );

        // Binds the vertex buffer that was previously allocated and filled with vertex data. This
        // consists of a 2D position and color for each vertex.
        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);

        // Binds the index buffer that was previously allocated and filled with data.
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT32
        );

        // Binds the descriptor set at the same index as this command buffer to the command buffer.
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,  // Specifies graphics pipeline usage
            self.data.pipeline_layout,  // `pipeline_layout` contains `descriptor_set_layout`
            0,  // Index of first parameter set
            &[self.data.descriptor_sets[image_index]],  // Array of descriptor sets to bind
            &[],  // Array of dynamic offsets (currently not used)
        );

        // Sets the slice of 64 bytes representing the model matrix to their allocated range at the
        // beginning of the push constants data.
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,  // Used by vertex shader
            0,  // Start position of data within push constant data
            model_bytes,
        );

        // Sets a single float representing the opacity of the model. This is position in offset
        // 64, directly after the 64 byte model matrix discussed above.
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,  // Used by fragment shader
            64,  // Start position of data within push constant data
            opacity_bytes,
        );

        // Draws the scene.
        self.device.cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);
        // Arguments 1 onwards are:
        //     - index_count – the number of vertices to draw
        //     - instance_count – 1 as this program does not use instanced rendering
        //     - first_index – index buffer offset
        //     - vertex_offset - vertex buffer buffer offset
        //     - first_instance – instanced rendering offset

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }
}

/// The Vulkan handles and associated properties used by the rest of the code.
#[derive(Clone, Debug, Default)]
struct AppData {
    surface: vk::SurfaceKHR,
    messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    msaa_samples: vk::SampleCountFlags,  // The MSAA sample counts supported by the physical device
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,  // Command pool used for initialization commands
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    color_image: vk::Image,  // Used to store the image that is then the source for MSAA
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
}

/// Creates a Vulkan "instance" from a `window` obtained from the OS, and a Vulkan `entry` point.
/// All created resources are saved in the fields of the `data` object passed.
unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData
) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Display a Viking room using Vulkan\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    // If the program is being built with debugging enabled, creates validation layers to display
    // error messages if the Vulkan interface is used incorrectly.
    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Creates a list of Vulkan extension functionality that the Vulkan instance must include to
    // allow program code to run.
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // // Lists the names of the extensions that will be passed to `create_instance` for debugging
    // // purposes.
    // println!("Window extensions: ");
    // extensions.iter().for_each(|ext| {
    //     let t = vulkanalia::vk::ExtensionName::from_ptr(*ext);
    //     println!("    Value at location {:p} is '{:?}'", ext, t);
    //
    // });

    // If running on MacOS version 1.3.216 or newer, enables the extensions and flags required to
    // run Vulkan on MacOS.
    let flags = if
            cfg!(target_os = "macos") &&
            entry.version()? >= PORTABILITY_MACOS_VERSION
        {
            info!("Enabling extensions for macOS portability.");
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

    // Creates a new Vulkan instance with the extensions and validation layers that are setup
    // above. A flag is also passed if running on a version of MacOS that requires extensions to
    // run Vulkan applications.
    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    // If validation layers are required (i.e., to display error messages for bad Vulkan API
    // usage), points to the callback function that will output them.
    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;
    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

/// Callback function that processes and displays debugging information received from the Vulkan
/// validation layers.
//
// Declared as an external system function to allow Vulkan to call it.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

/// Used to report missing functionality relating to physical device features.
#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

/// Returns the first physical graphics device that has a queue that meets the features needed
/// by the rest of this program.
unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

/// Returns a `Result` indicating whether the physical graphics device passed implements
/// the features needed by the rest of this program. These include the ability for its queue
/// families to run graphics commands and display to a surface; for its device extensions to
/// implement swapchains; and for anisotropic filtering to be available.
unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }

    Ok(())
}

/// Returns a `Result` indicating whether the physical graphics device passed implements
/// the device extensions required by the rest of this program. Specifically, the device needs to
/// implement all the extensions listed in the `DEVICE_EXTENSIONS` constant.
unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions.")))
    }
}

/// Stores information about the properties of a queue family of a physical graphics device.
#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,  // Ability to run graphics commands
    present: u32,  // Ability to present to a surface (i.e., display to a window)
}

impl QueueFamilyIndices {
    /// Determines if the physical graphics card passed has queue families whose queues accept
    /// graphics commands, and can present to a surface (i.e., display to a window). If so, the
    /// queue families' graphics and present properties are returned. If not, an error is returned.
    /// Note that it's possible to end up with one queue family that implements graphics commands
    /// and a different queue family that implements presentation (though unlikely).
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance
            .get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, _) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                data.surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}

/// Creates and returns a new logical device for the physical device set in the `data` object
/// passed. The logical device is created with graphics and presentation queues. The anisotropic
/// filtering feature is enabled.
unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];  // Priority for this queue doesn't matter as there's only one
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    // Enable the device extensions we need, e.g., swapchain support.
    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Enable Vulkan SDK portability if running on MacOS version 1.3.216 or newer.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        // Enable sample shading to allow smoothing of high-contrast edges _within_ a texture, not
        // just around the edges of an image.
        .sample_rate_shading(true);

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Last parameter for the creation of the graphics and presentation queues is the queue index.
    // This is 0 as only one queue is created (which is typical).
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

/// Holds the features supported by a particular swapchain, namely:
///     - capabilities, e.g., min/max # of images, current extent (resolution), min/max extents,
///       supported transforms, and supported usage flags,
///     - formats, e.g., mapping of RGBA to bits, and color space,
///     - presentation modes, defining how to handle multiple images queued for display.
#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    /// Returns the swapchain features supported by the physical device and surface passed.
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(
                    physical_device, data.surface)?,
        })
    }
}

/// Returns the best surface format from the given list of formats. The surface format consists of:
///     - the format itself - defining the color channels (e.g., red, green, blue, and/or alpha),
///     - the number of bits for each channel,
///     - the channel ordering,
///     - how to interpret the channel data, (e.g., as per the sRGB spec); and
///     - the color space.
///
/// If the standard 8-bit sRGB format is available for the sRGB color space, it is returned.
/// Otherwise, the first surface format in the list passed is returned.
//
// Production code should probably not just select the first item in the list passed, but instead
// more carefully pick which format to use in the unlikely event sRGB is unavailable.
fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

/// Returns the best presentation mode from the given list of possibilities. There are only four,
/// of which only FIFO is guaranteed to always be available. MAILBOX is chosen if it's available
/// as this overwrites any queued images with new ones, ensuring the latest image is always the
/// one displayed when it's time to send an image to the display. This is less energy efficient as
/// it results in throwing work away, unlike FIFO which displays all images in turn as they are
/// placed into the queue.
fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

/// Returns the best `Extent2D`, which defines the resolution of the swapchain images (which is
/// almost always the resolution of the display). The `capabilities` passed into this function
/// include a `current_extent` field that contains either: the current resolution of the surface;
/// or a special resolution where both width and height are u32::MAX that indicates that the
/// surface size is determined from the resolution of swapchain images. In the latter case, the
/// `min_image_extent` and `max_image_extent` fields define the allowed ranges for width and
/// height, and the return value is the current window resolution with width and height clamped to
/// these ranges.
fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |minimum: u32, maximum: u32, v: u32| minimum.max(maximum.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width,
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height,
            ))
            .build()
    }
}

/// Creates a swapchain for the given physical `device`.
unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;
    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    // Creates one more image than the minimum required by the swapchain implementation to reduce
    // time spent idling while waiting for an image to become available. However, if this exceeds
    // the maximum number of images, conform to that.
    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    // In the unlikely situation where graphics are created on one queue and presented on a
    // different queue, sets the CONCURRENT sharing mode so swapchain images can be accessed from
    // both. Otherwise, uses EXCLUSIVE mode.
    //
    // High-performance production code would probably use EXCLUSIVE mode for the first
    // case and explicitly transfer ownership of swapchain images back and forth between queues.
    // The tutorial avoids tackling this complexity.
    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)  // > 1 only if creating dual images, e.g., VR
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)  // Indicates rendering direct to image
        .image_sharing_mode(image_sharing_mode) // Concurrent or exclusive sharing
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)  // No rotation or flipping
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)  // No window transparency
        .present_mode(present_mode)
        .clipped(true)  // Don't care about the contents of windows obscuring ours
        .old_swapchain(vk::SwapchainKHR::null());  // See below
        // The last option must be used if a new swapchain is created to obsolete an old one, e.g.,
        // because the window is resized. This program deletes the current swapchain and creates a
        // new one in this situation, so this field is always null.

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;

    Ok(())
}

/// Creates an "image view" for every swapchain image in the `AppData` passed. This defines
/// attributes such as the color mapping and the part of the image to be written to.
unsafe fn create_swapchain_image_views(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| create_image_view(
            device,
            *i,
            data.swapchain_format,
            vk::ImageAspectFlags::COLOR,
            1,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// Creates a "texture image view" for the `data.texture_image`. The format chosen is 8-bit RGBA
/// in the SRGB color space.
unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    Ok(())
}

/// Creates and returns an image view for `image` that is in `format`. The former specifies how
/// image data is interpreted and allows images to be treated as 1D textures, 2D textures,
/// 3D textures, and cube maps. The latter specifies formats such as `vk::Format::R8G8B8A8_SRGB`.
/// `aspects` defines whether the image is to be used for COLOR or DEPTH. The number of
/// mip levels is passed as a parameter to allow mipmapping.
unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    // A mostly basic setup with no mipmapping or multiple layers. The latter is useful for VR.
    // However, multiple mip levels are set to enable mipmapping.
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    // The `view_type` and `format` fields specify how image data is interpreted. The former
    // allows images to be treated as 1D textures, 2D textures, 3D textures, and cube maps.
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

/// Creates a pipeline for the given device based on the bytecode SPIR-V vertex and fragment
/// shaders compiled separately from source code in the shaders directory of this project.
unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");  // The name of the fn in the shader that is called to run the shader

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");  // The name of the fn in the shader that is called to run the shader

    // Creates binding and attribute descriptions that define how data defined in this program is
    // passed to the vertex shader.
    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // The tutorial specifies a triangle list as this is the format output from the code that reads
    // the model file, but several other ways of representing the vertex data are are available
    // and may be more appropriate for other applications.
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Creates a basic viewport that extends from the very top-left corner (at 0.0, 0.0) to the
    // size of the images in the swapchain. The depth range is defined as 0.0 to 1.0, which is
    // typical for Vulkan applications.
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // Defines the scissor region to be the entire framebuffer as no cropping is required.
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Builds a rasterizer to take the output of the vertex shader stage as input and convert it
    // to fragments.
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)  // Useful if doing shadow mapping, but not for our simple case
        .rasterizer_discard_enable(false)  // Disables the rasterizer if true
        .polygon_mode(vk::PolygonMode::FILL)  // See below
        .line_width(1.0)  // Thickness of lines in units of fragments
        .cull_mode(vk::CullModeFlags::BACK)  // Cull back faces
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)  // Culling clockwise or CCW faces?
        .depth_bias_enable(false);  // Other values maybe useful for shadow mapping
    // PolygonMode can also be LINE or POINT, which renders vertices as wireframes or just dots.
    // Maybe useful for debugging. However, anything but FILL requires additional GPU extensions
    // to be enabled.

    // Enables anti-aliasing and sample shading. The former smooths jagged edges around fragments
    // and the latter smooths high-contrast jagged edges within fragments.
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        // Minimum fraction for sample shading; closer to one is smoother.
        .min_sample_shading(0.2)
        .rasterization_samples(data.msaa_samples);

    // Configures the depth buffer. Fragments with a higher depth value than the current depth
    // value in the buffer are discarded. Fragments with a lower value are rendered and their
    // depth value is written to the buffer.
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)  // Discard fragments further than current depth buffer value
        .depth_write_enable(true)  // Enables writing new fragments' depths to buffer
        .depth_compare_op(vk::CompareOp::LESS)  // Lower depth value = closer to camera (standard)
        .depth_bounds_test_enable(false)  // Disable discard of fragments outside following bounds
        // .min_depth_bounds(0.0)  // Unused for this program
        // .max_depth_bounds(1.0)  // Unused for this program
        .stencil_test_enable(false);  // Not using stencils

    // Does not enable color blending, so a newly calculated fragment color overwrites any color
    // already in the framebuffer (unless logical blending is enabled - see next code block).
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)  // Enable blending to allow fragment shader to change opacity
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)  // Optional
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO) // Optional
        .alpha_blend_op(vk::BlendOp::ADD);             // Optional

    // Does not enable logical color blending, which allows a bitwise combination of the existing
    // color for a fragment in the framebuffer and a newly calculated value. If logical color
    // blending is enabled, color blending is disabled (see previous code block).
    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // As no color blending functionality is enabled, a newly calculated fragment color value
    // simply overwrites any existing value for a fragment in the framebuffer.

    // // The following commented code block can be used to dynamically alter some state without
    // // needing to recreate the pipeline. It is optional and not currently needed.
    // let dynamic_states = &[
    //     vk::DynamicState::VIEWPORT,
    //     vk::DynamicState::LINE_WIDTH,
    // ];
    //
    // let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
    //     .dynamic_states(dynamic_states);

    // Defines a push constant range that can be accessed by the vertex shader. Updating push
    // constants is faster than updating UBOs, so this range is used to pass the model matrix to
    // the vertex shader as this changes per model every frame (as the model is rotated). Push
    // constants have a very limited size, so are too small to also pass the view and projection
    // matrices.
    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)  // Data offset of vertex data within entire push constant data set
        .size(64);  // 4x4 matrix of 4 byte floats

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(64)  // Starts at offset 64, after the vertex push constants created above
        .size(4);  // One 4 byte float

    // Pipeline layouts allow data to be passed to "uniform" declarations in shader code (which
    // are used for both push constants and UBOs). These are effectively global variables. This
    // technique is commonly used to pass the transformation matrix to the vertex shader, or to
    // create texture samplers in the fragment shader.
    // Builds a pipeline layout with the existing UBO (uniform buffer object) so that UBO is
    // available in the pipeline stages, and push constant ranges for the model matrix and a value
    // controlling opacity.
    let set_layouts = &[data.descriptor_set_layout];
    let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);
    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Using all the above setup finally creates the pipeline.
    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(), &[info], None)?.0[0];

    // Now vertex and fragment shader functionality has been incorporated into the pipeline, the
    // shader modules used to prepare them can be safely deleted.
    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

/// Wraps the given shader bytecode in a `vk::ShaderModule` object which is then returned. As part
/// of this process, the binary SPIR-V file read from disk is translated to the bytecode format
/// required by Vulkan.
unsafe fn create_shader_module(
    device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule> {
    // Converts the bytecode from u8 bytes (the output of the glslc command) to u32 bytes (as
    // required by the `ShaderModule` object.
    let bytecode = Vec::<u8>::from(bytecode);
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(anyhow!("Shader bytecode is not properly aligned."));
    }

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.len())
        .code(code);

    Ok(device.create_shader_module(&info, None)?)
}

/// Creates a render pass. This specifies the initial and final states to use for images, creates
/// color attachments, creates subpasses and dependencies between subpasses.
unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(data.msaa_samples)  // Enable MSAA
        .load_op(vk::AttachmentLoadOp::CLEAR)  // Clear buffer contents before buffer use
        .store_op(vk::AttachmentStoreOp::STORE)  // Allow buffer to be read after rendering
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)  // Not using stencils
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)  // Not using stencils
        .initial_layout(vk::ImageLayout::UNDEFINED)  // See below
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);  // See below
    // Images transition between two `ImageLayout`s - the one in force when the image becomes
    // available for use, and the one transitioned to when rendering to the image is complete and
    // it is ready for presentation. The _initial_ image layout is unimportant because this program
    // explicitly clears the contents of each image before rendering is performed. The _final_
    // image layout is suitable for an MSAA image, which cannot be displayed directly.

    // Creates a depth stencil attachment to store depth data during the rendering phase only. This
    // is ephemeral data, so we don't care what's in the buffer beforehand and don't care about
    // storing the data after the rendering's complete.
    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, data)?)  // Format must be the same as the depth image
        .samples(data.msaa_samples)  // Enable MSAA
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)  // Don't store depth data after use
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)  // Don't care about initial contents
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Creates an MSAA attachment to store the MSAA version of the final image. This is "resolved"
    // from an MSAA image layout that cannot be displayed directly, to a final image that can.
    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // SUBPASSES

    // Creates a color attachment reference to the only color reference created earlier (which is
    // at index position 0).
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // Creates an attachment for depth testing.
    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Creates a resolve attachment to resolve the MSAA image to a format that can be displayed.
    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachment_ref];

    // Builds a subpass that references the color, depth and resolve (i.e., MSAA) attachments
    // created above. Multiple subpasses can be created for a rendering pass to efficiently create
    // an image that requires multiple passes, but this program only needs a single subpass. A
    // subpass can have multiple color attachments, but only one depth (& stencil) attachment as it
    // doesn't make sense to have multiple depth buffers.
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)  // Initial image that can be used as an MSAA source
        .depth_stencil_attachment(&depth_stencil_attachment_ref)  // Depth info used during render
        .resolve_attachments(resolve_attachments);  // MSAA source to final image layout

    // Defines dependencies between subpasses so that they are executed at the correct times.
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)  // Source subpass - EXTERNAL occurs before all others
        .dst_subpass(0)  // The subclass created above is at index 0 as it's the only one defined
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)  // Wait on color & fragment stages
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE  // Later code waits on color …
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);  // … & depth writing completion

    // Creates arrays of "attachments", "subpasses", and dependencies, and populates them with the
    // single `color_attachment`, `subpass`, and `dependency` created above.
    let attachments = &[
        color_attachment,
        depth_stencil_attachment,
        color_resolve_attachment,
    ];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    // Creates the actual render pass.
    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

/// Creates a framebuffer object for each swapchain image view in the `AppData` structure passed
/// and adds it to the `framebuffers` field of `AppData`.
unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            // Create a framebuffer that references the common MSAA image (named
            // "color_image_view"), the common depth attachment and the swapchain image view
            // created for this framebuffer.
            let attachments = &[data.color_image_view, data.depth_image_view, *i];

            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)

        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

/// Creates a single new command pool for "allocating command buffers during initialization", and
/// one new command pool for each framebuffer. The `AppData` object passed is updated to point to
/// the new pools.
unsafe fn create_command_pools(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Creates a single command pool for "allocating command buffers during initialization".
    data.command_pool = create_command_pool(instance, device, data)?;

    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

/// Returns a newly created command pool.
unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)  // Optional hint. Buffers are short-lived
        .queue_family_index(indices.graphics);

    Ok(device.create_command_pool(&info, None)?)
}

/// Creates one primary command buffer and an empty vector of to hold secondary command buffers
/// for every swapchain image defined in the `AppData` object passed. `AppData` is updated to
/// point to the new command buffers.
unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let num_images = data.swapchain_images.len();
    for image_index in 0..num_images {
        // Creates one command buffer for every swapchain image in `AppData`.
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    // Creates the secondary command buffers.
    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    Ok(())
}

/// Creates two semaphore objects and one fence for every frame that could be in flight at once
/// (as defined in the `MAX_FRAMES_IN_FLIGHT` constant). One semaphone signals when an image has
/// been acquired and rendering can begin, and the other signals when rendering is complete and
/// the image is ready to be displayed. The fence is used to ensure that the semaphores associated
/// with a particular frame are not reused before that frame has been displayed by pausing the
/// start of the creation of the next frame if we've reached `MAX_FRAMES_IN_FLIGHT`.
unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);  // Creates fences in the signaled state

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
    }

    // Creates a placeholder fence for each swapchain image that indicates if the image is
    // currently in use by an in-flight frame. The fence of an in-flight frame is placed in
    // this structure for the swapchain image in use, thereby mapping frames and images while
    // they are in use.
    data.images_in_flight = data.swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

/// The 3D position, color and texture coordinates of a single vertex, stored in C representation
/// so it can be passed to the vertex shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    /// Returns a new `Vertex` from the 3D position, color passed and 2D texture coordinate.
    /// Defined as `const` to allow it to be used to define constants.
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self { pos, color, tex_coord }
    }

    /// Returns a "binding description" that specifies how data will be passed to the vertex
    /// shader.
    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)  // The index of this binding (0 as this program only needs one binding)
            .stride(size_of::<Vertex>() as u32)  // Size of each vertex entry in bytes
            .input_rate(vk::VertexInputRate::VERTEX)  // Each entry is 1 vertex, not 1 instance
            .build()
    }

    /// Returns a 3-element array containing the attribute descriptions for the 2D position, color
    /// and texture components of the vertex data.
    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)  // The index of this binding (0 as this program only needs one binding)
            .location(0)  // Location 0 of each vertex entry is the 2D position
            .format(vk::Format::R32G32B32_SFLOAT)  // These are X, Y & Z coords, not color
            .offset(0)  // Position data is at very beginning of each vertex entry
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)  // The index of this binding (0 as this program only needs one binding)
            .location(1)  // Location 1 of each vertex entry is the RGB color
            .format(vk::Format::R32G32B32_SFLOAT)  // Color format - 3x 32 bit signed floats
            .offset(size_of::<Vec3>() as u32)  // Color data begins after the position data
            .build();

        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)  // The index of this binding (0 as this program only needs one binding)
            .location(2)  // Location 2 of each vertex entry is the texture coordinate
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();

        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    /// Returns true if the position, color _and_ texture coordinates of `self` match their
    /// respective fields in `other`.
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    /// Returns a hash of `self` based on X, Y, Z position coordinates; R, G, B color values; and
    /// U and V texture coordinates.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

/// Creates a buffer that will be used to hold vertex data, maps its memory to RAM, and copies the
/// vertex data from `AppData` into it. The data is loaded via a staging buffer so that it can be
/// stored in higher performance VRAM, if available.
unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

    // Creates a buffer. The TRANSFER_SRC usage ensures the buffer can be used as the source in a
    // memory transfer operation. The HOST_VISIBLE property ensures the buffer can be accessed by
    // the CPU to allow initial vertex data to be copied in. HOST_COHERENT ensures reads and writes
    // see a consistent via of the data in the buffer. This is the simpler, but potentially slower,
    // alternative to flushing after each set of writes and "invalidating" buffer data before each
    // read.

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Maps standard RAM to the vertex buffer to allow the CPU to write to it.
    let memory = device.map_memory(
        staging_buffer_memory,
        0,  // Offset
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    // Copies vertex data stored in `AppData` to the mapped vertex buffer memory.
    memcpy(data.vertices.as_ptr(), memory.cast(), data.vertices.len());

    // Unmaps the vertex buffer memory from RAM as all vertex data is now present in the staging
    // buffer.
    device.unmap_memory(staging_buffer_memory);

    // Creates a buffer. The TRANSFER_DST usage ensures the buffer can be used as the destination
    // in a memory transfer operation. The VERTEX_BUFFER usage ensures the buffer can be used as a
    // vertex buffer. The DEVICE_LOCAL property ensures the device means the memory is optimized
    // for use by the graphics device, typically because it is VRAM on the device itself.
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    // Copies data from the staging buffer to the vertex buffer.
    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    // Destroys the staging buffer and frees the memory associated with it.
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

/// Creates a buffer that will be used to hold vertex index data, maps its memory to RAM, and
/// copies the index data from `AppData` into it. The data is loaded via a staging
/// buffer so that it can be stored in higher performance VRAM, if available.
//
// Note that the implementation is very similar to `create_vertex_buffer()`, so refer to the
// comments for that code.
unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let size = (size_of::<u32>() * data.indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    // HOST_COHERENT ensures reads and writes see a consistent via of the data in the buffer
    // without explicit flushes. HOST_VISIBLE ensures the CPU can access the buffer.

    let memory = device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

/// Creates a buffer, allocates memory suitable for the requested `usage` and with the requested
/// `properties`, binds the memory to the buffer and returns the resulting buffer and allocated
/// memory.
unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);  // Buffer only used from the graphics queue

    let buffer = device.create_buffer(&buffer_info, None)?;

    // Determines the amount of memory that the buffer requires, its offset in memory and the
    // memory types that it supports. This data is returned in a `vk::MemoryRequirements` struct.
    let requirements = device.get_buffer_memory_requirements(buffer);

    // Builds the structure needed to allocate memory that requests memory based on the size
    // needed by the vertex buffer, and a type that is:
    //     - suitable for the vertex buffer,
    //     - supported by the physical device, *and*
    //     - supports the requested `properties` passed to this function.
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    // Allocates the memory for the vertex buffer.
    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    // Binds the newly allocated memory to the vertex buffer.
    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

/// Returns a memory type supported by the physical device that meets the given `properties` and
/// `requirements`.
unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    // Gets the physical device's memory heap details (such as whether it is VRAM), and the
    // memory types available within each memory heap.
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    // Ignores the memory heaps and compares the requested memory `properties` and memory
    // 'requirements` passed to this function with the physical device's capabilities.
    (0..memory.memory_type_count)
        .find(|i| {
            // Is this memory_type set in the desired `requirements` passed to this function?
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;

            let memory_type = memory.memory_types[*i as usize];

            // If so, and *every* desired property flag passed in `properties` is present in the
            // physical device's property flags associated with this memory type, the memory
            // type is a match. The "contains" call is defined by Vulkanalia for the property
            // flags type, which is a bit field.
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

/// Copies the contents of the `source` buffer to the `destination` buffer and returns once the
/// copy completes. This is achieved by creating a command buffer for one-time commands and using
/// it to record the copy command.
unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    // Records the buffer copy command.
    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    // Closes and submits buffer containing the copy command, and returns only once the copy
    // completes.
    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

/// Creates and returns an open command buffer ready to record additional commands. The buffer is
/// created for use as a one-time submission, and the `end_single_time_commands` function must be
/// called to end recording.
unsafe fn begin_single_time_commands(
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    // Allocates a new command buffer to record commands. This uses the existing command pool, but
    // in production code it is better to create a command pool that is only used for one time
    // command buffers and create it with the vk::CommandPoolCreateFlags::TRANSIENT to indicate its
    // purpose to benefit from Vulkan optimizations.
    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);  // Command buffer is only used once

    // Begins recording commands.
    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

/// Closes and submits a one-time command buffer previously created with
/// `begin_single_time_commands` and which has been used to record commands to be executed once,
/// then waits until the commands complete.
unsafe fn end_single_time_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    // Ends recording.
    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers);

    // Submits the buffer for execution and waits until all commands complete.
    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    // Frees the command buffer.
    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

/// Holds the view and projection matrices that will be passed to shaders via a UBO (uniform
/// buffer object).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    view: Mat4,
    proj: Mat4,
}

/// Builds two descriptor set layouts. The first refers to a `UniformBufferObject` that is used
/// to pass global data to vertex shaders. The second refers to a combined image sampler, which
/// is used by fragment shaders to access texture data.
unsafe fn create_descriptor_set_layout(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Creates a binding to allow vertex shaders to access UBOs (uniform buffer objects).
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)  // Number of description sets defined in array
        .stage_flags(vk::ShaderStageFlags::VERTEX);  // Only using descriptor in vertex stage

    // Creates a binding for the combined image sampler, to be used by fragment shaders to access
    // texture data.
    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);  // Only using in fragment shader

    // Independently creates bindings for UBOs and the sampler.
    let bindings = &[ubo_binding, sampler_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

/// Creates uniform buffers to pass matrices to shaders each frame. The number of buffers created
/// is the same as the number of framebuffer images to allow a frame in the process of being
/// rendered to read from its own uniform buffer while data is being written to a different
/// uniform buffer to be used for the next frame.
//
// A staging buffer is not used because it is likely the uniform data will change every frame, so
// the overhead of staging will hurt performance rather than improve it.
unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,  // This buffer is for a UBO
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        // HOST_COHERENT ensures reads and writes see a consistent via of the data in the buffer.
        // This is the simpler, but potentially slower, alternative to flushing after each set of
        // writes and "invalidating" buffer data before each read.

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

/// Creates two new descriptor pools to enable shaders to read resources, namely:
///     - combined image samplers to enable texture data to be read; and
///     - UBOs (uniform buffer objects), to enable the view and projection matrices to be read.
unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    // Determines the size required to store a pool of combined image sampler descriptors where
    // there is one sampler for each swapchain image.
    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    // Determines the size required to store a pool of UBO descriptors where there is one UBO for
    // each swapchain image.
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    // Creates two new descriptor pools: one for pointers to UBOs; and the other for pointers to
    // combined image samplers. Each pool will have one descriptor for each swapchain image.
    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

/// Creates descriptor sets and associates them with UBOs (uniform buffer objects) and combined
/// texture samplers. One UBO descriptor set and one sampler is created for each swapchain image.
unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);  // Also implicitly sets `descriptor_set_count` to the array length

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    // Initializes each descriptor set to an associated uniform buffer object or image sampler
    for i in 0..data.swapchain_images.len() {
        // Specifies which buffer, and the offset within it, contains the descriptor data for each
        // UBO.
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)  // Must match the binding chosen in `create_descriptor_set_layout`
            .dst_array_element(0)  // Offset to element in buffer_info array
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        // Specifies which buffer, and the offset within it, contains the descriptor data for each
        // combined image sampler.
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)  // Must match the binding chosen in `create_descriptor_set_layout`
            .dst_array_element(0)  // Offset to element in buffer_info array
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        // Updates the descriptor sets. This can be done with either a set of `WriteDescriptorSet`s
        // or `CopyDescriptorSet`s. This code uses the former, which is why the second parameter
        // is an empty array.
        device.update_descriptor_sets(
            &[ubo_write, sampler_write],
            &[] as &[vk::CopyDescriptorSet],  // Not copying, so copy source array is empty
        );
    }

    Ok(())
}

/// Reads a PNG image from an OS file with a hard-coded filename, and loads it into a texture on
/// VRAM via a staging buffer.
/// Note: the PNG image must:
///     - have an alpha channel (though the "png" crate now includes an option to add an
///             alpha channel with png >= 0.17.10); and
///     - be in the SRGBA color space.
unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let image = File::open("resources/viking_room.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;
    let mut pixels = vec![0;  reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();

    // Determines the number of mip levels required for this image. An image is required for
    // each halving of both width and height of the image until a mip of size 1x1 is reached.
    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    // Sanity checks that the loaded image is suitable for use with this program.
    if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
        panic!("Invalid texture image.");
    }

    // Creates a staging buffer that: the CPU can write to; and which can be used as the source of
    // data for transfers.
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    // HOST_COHERENT ensures reads and writes see a consistent via of the data in the buffer
    // without explicit flushes. HOST_VISIBLE ensures the CPU can access the buffer.

    // Maps the staging buffer to RAM so data can be written to it.
    let memory = device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    // All data has been copied, so unmap the staging buffer from RAM.
    device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,  // MSAA sample count
        vk::Format::R8G8B8A8_SRGB,  // Texel and pixel image formats must be the same
        vk::ImageTiling::OPTIMAL,  // Pixels are laid out in whatever way is optimal
        vk::ImageUsageFlags::SAMPLED  // See below
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    // The usage flags in turn allow the image to be: sampled by the shader to color the
    // graphical mesh; used as a destination buffer and used as a source buffer. The latter is
    // needed for the mipmapping functionality, where new mip levels are created by taking the
    // original texture contents, reducing their resolution and writing the result to a different
    // mip level in the same image.

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // Convert the image's layout from the "UNDEFINED" that the buffer data was created using to
    // the opaque "optimal" layout the physical device prefers.
    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,  // Must match the image layout of the source image
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;

    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    Ok(())
}

// Creates an image object suitable for storing image pixels. This function returns a tuple of
// image object and its associated block of memory, but does not actually copy any image pixels
// into either. This code creates different types of images depending on the `usage` and
// `properties` parameters. The program uses this code to create an image to store a texture,
// images for the swapchain and an image to use as a depth buffer.
unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,  // MSAA sample count
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)  // Image data is not an array
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)  // Starting contents are not usable
        .usage(usage)
        .samples(samples)  // MSAA sample count
        .sharing_mode(vk::SharingMode::EXCLUSIVE);  // Only used by one queue family

    // Creates the image. If using less common image formats than SRGB with alpha, it's possible
    // the call will fail and then require runtime image conversion to a supported format.
    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

/// Transitions `image` from `old_layout` to `new_layout`. Only two transitions are supported by
/// this function:
///     - from UNDEFINED to TRANSFER_DST_OPTIMAL: typically used to transition newly loaded image
///       data to the hardware device's preferred layout.
///     - TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL: to transition an image layout to an
///       optimal layout that is best for shaders to read.
unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    // Determines if the requested transition between old and new layouts is supported by this
    // function, and return an error if not. If supported, set appropriate values for:
    //     - the source and destination access masks, which define the resources and type of access
    //       that the operation has (which partly defines when it can run); and
    //     - the source and destination pipeline stage masks, which define the pipeline stages when
    //       the operation can run.
    let (
        src_access_mask,
        dst_access_mask,
        src_stage_mask,
        dst_stage_mask,
    ) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),  // Do not wait on any previous action
            vk::AccessFlags::TRANSFER_WRITE,  //
            vk::PipelineStageFlags::TOP_OF_PIPE,  // Very first pipeline stage
            vk::PipelineStageFlags::TRANSFER,  // Pseudo pipeline stage that is used for transfers
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,  // The earliest pipeline stage the operation can run
            vk::PipelineStageFlags::FRAGMENT_SHADER,  // The latest pipeline stage it can run
        ),
        _ => return Err(anyhow!("Unsupported image layout transition!")),
    };

    let command_buffer = begin_single_time_commands(device, data)?;

    // Defines the subresources of the image to be changed by the image memory barrier layout
    // operation.
    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)  // Only affect the color properties of the image
        .base_mip_level(0)
        .level_count(mip_levels)  // Create one level for every mip level requested
        .base_array_layer(0)  // Index of layer
        .layer_count(1);  // Only one layer is defined in the image

    // Use an image memory barrier to convert the layout (i.e., the format) of the image.
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)  // Do not transfer ownership …
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)  // … between queue families
        .image(image)  // Image on which to apply the memory barrier
        .subresource_range(subresource)  // Part of image to affect
        .src_access_mask(src_access_mask)  // Mask of accesses allowed to the source image
        .dst_access_mask(dst_access_mask); // Mask of accesses allowed to tmp dest. image layout

    // Add the image memory barrier to the command pipeline so it will be executed.
    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,  // Mask of accesses allowed to temporary source image layout
        dst_stage_mask,  // Mask of accesses allowed to final image layout
        vk::DependencyFlags::empty(),  // Wait until layout change is fully complete
        &[] as &[vk::MemoryBarrier],  // Not using a memory barrier
        &[] as &[vk::BufferMemoryBarrier],  // Not using a buffer memory barrier
        &[barrier],  // Specify the image memory barrier created above
    );
    // The second parameter is the stage (or stages) to execute before the barrier. The third
    // parameter are the stages that wait for the barrier to complete. The set of pipeline stage
    // flags that are permitted depend on the type of barrier being implemented.

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

/// Copies the contents of `buffer` to `image`.
unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    // Defines the subresources of the image to copy.
    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)  // Byte offset of the start of the pixel values
        .buffer_row_length(0)  // Pixel layout. 0 = usual packed layout with no padding
        .buffer_image_height(0)  // Ditto
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D { width, height, depth: 1 });
    // The last 3 fields define the part of the image the pixel data will be written to.

    let command_buffer = begin_single_time_commands(device, data)?;

    // Copies the pixel data in the `buffer` passed to the `image` passed. The `buffer` data must
    // be in the *optimal* layout format (which is whatever opaque format that the physical device
    // prefers), so a conversion to this format must be performed prior to the call to this
    // function.
    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,  // Source image data
        image,  // Destination for image data
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,  // *Source* layout format
        &[region],  // Can copy the same data to multiple regions, but only 1 region for this code
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

// Creates a texture sampler and adds it to `AppData`. The sampler is used by shaders to read
// data from image views, and applies filters such as anisotropic filtering to improve image
// quality.
unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)  // Magnified (i.e., oversampled) texels - see below
        .min_filter(vk::Filter::LINEAR)  // Minified (i.e., undersampled) texels - see below
        .address_mode_u(vk::SamplerAddressMode::REPEAT)  // Repeat texture in u direction [x]
        .address_mode_v(vk::SamplerAddressMode::REPEAT)  // Repeat texture in v direction [y]
        .address_mode_w(vk::SamplerAddressMode::REPEAT)  // Repeat texture in w direction [z]
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)  // Used for samples outside tex. border
        .unnormalized_coordinates(false)  // False = ranges are 0..1, true = 0..width/height
        .compare_enable(false)  // Comparison disabled
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)  // Blend two mip levels at the boundaries
        .min_lod(0.0)  // Minimum level of detail
        .max_lod(data.mip_levels as f32)  // Maximum level of detail
        .mip_lod_bias(0.0);  // Bias applied to the level of detail to fine-tune it
    // *Oversampling* is where the same texel is used for multiple pixels because there are more
    // fragments than texels. Without filtering, it leads to a blocky Minecraft-like appearance.
    // The solution is to sample multiple texels and average the result, an example being a
    // bilinear filter.
    // *Undersampling* is where one pixel maps to multiple texels and the nearest texel is
    // selected. It causes blurriness if the texture is a high frequency pattern, such as a
    // checkerboard pattern viewed from certain angles. The solution is to sample several nearby
    // texels, e.g., using an anisotropic filter, which is the approach taken here.

    data.texture_sampler = device.create_sampler(&info, None)?;

    Ok(())
}

// Creates a depth image buffer, memory for it, and a view into it. Only one depth image buffer is
// created as it is only required during rendering, and this program only renders one frame at a
// time.
unsafe fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Chooses the best depth format for this program's requirements.
    let format = get_depth_format(instance, data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,  // Depth buffer needs same dimensions as swapchain extents
        data.swapchain_extent.height,
        1,  // Count of mip levels
        data.msaa_samples,  // MSAA sample count
        format,
        vk::ImageTiling::OPTIMAL,  // Use hardware-specific format that gives best performance
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,  // Want stencil containing depth info
        vk::MemoryPropertyFlags::DEVICE_LOCAL,  // Use VRAM as other code doesn't access depth info
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;

    // Creates an image view into the newly created depth image so it can be used.
    data.depth_image_view = create_image_view(
        device,
        data.depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
        1,  // Count of mip levels
    )?;

    Ok(())
}

/// Returns the best depth format from the list of `candidates` for the `tiling` mode passed. In
/// this context, 'best' refers to the depth usage made by this program. Most hardware supports
/// 32-bit signed floats (vk::Format::D32_SFLOAT), but some offer 24- or 32-bit floats with an
/// 8-bit stencil component.
///
/// The first candidate in the `candidates` array that is suitable is returned.
unsafe fn get_supported_format(
    instance: &Instance,
    data: &AppData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            // Populates `properties` fields with the features supported by the physical device:
            //     - linear_tiling_features – use cases supported with linear tiling
            //     - optimal_tiling_features – use cases supported with optimal tiling
            //     - buffer_features – use cases supported for buffers [never used by this program]
            let properties = instance.get_physical_device_format_properties(
                data.physical_device,
                *f,
            );

            // Checks that all requested `features` are available for the requested `tiling` mode.
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

/// Chooses the best depth buffer format from a hard-coded array of formats. OPTIMAL image tiling
/// is requested to allow the physical device to use whatever hardware-specific, opaque format it
/// wishes for best performance. If depth information needed to be shared with other parts of the
/// program, LINEAR image tiling may be required instead.
unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,  // 32-bit signed float
        vk::Format::D32_SFLOAT_S8_UINT,  // 32-bit signed float plus 8-bit stencil
        vk::Format::D24_UNORM_S8_UINT,  // 24-bit unsigned normalized float plus 8-bit stencil
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,  // Okay for device to store depth info any way it wishes
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,  // Required depth testing features
    )
}

/// Loads model data from a file and places the vertices and indices it contains into the `AppData`
/// object passed.
fn load_model(data: &mut AppData) -> Result<()> {
    let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);

    let (models, _) = tobj::load_obj_buf(  // Ignore material data loaded from file
        &mut reader,
        &tobj::LoadOptions { triangulate: true, ..Default::default() },  // Convert to triangles
        |_| Ok(Default::default()),  // Callback for materials. Not needed as not using materials
    )?;

    let mut unique_vertices = HashMap::new();

    // Iterate over every index in every model loaded from the file, appending each vertex and
    // index into the respective fields of the `AppData` object passed.
    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;  // 3 = X, Y, and Z values for each position
            let tex_coord_offset = (2 * index) as usize;  // 2 = U and V texture values

            let vertex = Vertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            };

            // Uses a `HashMap` to ensure that only unique vertices are stored. If the vertex just
            // created is already stored, the index of the previously stored copy is used rather
            // than creating a duplicate entry.
            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32);
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    Ok(())
}

/// Generates all mipmaps for `image`. The original `image` must be in `format`, have a resolution
/// of `width` and `height` and have `mip_levels` mip levels to store the mipmap data.
/// [I'm unsure why this data is passed rather than just passing `image` and having this function
/// determine the resolution and mip levels.]
//
// Production code normally never generates mipmapping data, but rather loads in pregenerated
// mipmaps from texture files.
unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    // Checks that the physical device supports linear blitting which this function relies on to
    // perform the scaling operations to generate successively lower resolution mip levels.
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!("Texture image format does not support linear blitting!"));
    }

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    // Creates an image memory barrier. It is mutable because it is reused for different purposes
    // and needs to be tweaked for each purpose.
    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    // `i` stores the *destination* mip level, so the *source* mip level is often referred
    // to within the loop as `i - 1`.
    for i in 1..mip_levels {
        // Reconfigures the memory barrier to transition the buffer layout to be optimized for
        // using as a source.
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        // Transitions the buffer layout. As this is done with a memory barrier, this will wait for
        // previous commands that are filling the source buffer to complete before running.
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Source mip level.
        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        // Destination mip level.
        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        // Creates a blit operation to reduce the resolution of the texture. The destination width
        // and height are set to half of the source, with a minimum value of 1.
        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,  // Must be 1 as texture is 2D
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,  // Must be 1 as texture is 2D
                },
            ])
            .dst_subresource(dst_subresource);

        // Performs the blit operation. `image` is specified as both the source and destination
        // which is okay because data is read and written to different mip levels.
        device.cmd_blit_image(
            command_buffer,
            image,  // Source
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,  // Destination
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,  // Enable interpolation during resolution reduction operation
        );

        // Reconfigures the memory barrier to change the layout of the newly created mip level
        // to be optimal for shader reading.
        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        // Performs the layout change. It waits on the current blit command to complete before
        // changing the layout. Sampling operations that occur later will wait until this layout
        // change is complete before reading data.
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        // Reduces the height and width of the current mip level, ready to be used as the
        // source dimensions in the next loop iteration.
        if mip_width > 1 {
            mip_width /= 2;
        }
        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    // This code is identical to that in the loop but is needed to transition the last (and
    // smallest) mip level.
    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

/// Returns the maximum multi-sample anti-aliasing (MSAA) count the chosen physical device
/// supports. The physical device is queried for all the MSAA counts it supports, which is
/// returned as a single value containing the bit flags for the supported levels. This function
/// then returns the maximum MSAA count from this list. If no MSAA count information is obtained,
/// the MSAA count returned is 1, which equates to no MSAA.
unsafe fn get_max_msaa_samples(
    instance: &Instance,
    data: &AppData,
) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(data.physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}

/// Creates an image to hold data to be used as an MSAA color source and stores the image, its
/// associated memory, and a view for the image in the `AppData` structure passed.
unsafe fn create_color_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let (color_image, color_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,  // An image cannot use both mipmapping and MSAA, so mip levels must be set to 1
        data.msaa_samples,
        data.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;

    data.color_image_view = create_image_view(
        device,
        data.color_image,
        data.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,  // No mipmapping
    )?;

    Ok(())
}
