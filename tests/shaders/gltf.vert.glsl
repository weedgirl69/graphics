#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in mat3x4 in_instance_transform;
layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_view_direction;

layout(binding = 0) uniform FrameUniforms {
    mat4 view_projection;
    vec3 camera_position;
};

void main() {
  vec3 position = vec4(in_position * .01, 1.0) * in_instance_transform;

  out_normal = in_normal * mat3(in_instance_transform);
  out_view_direction = camera_position - position;

  gl_Position = vec4(position, 1) * view_projection;
}