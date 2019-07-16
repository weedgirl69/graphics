#version 450

layout(location=0)in vec3 in_position;
layout(location=1)in vec3 in_normal;
layout(location=2)in mat3x4 in_instance_transform;
layout(location=0)out vec3 out_normal;
layout(location=1)out vec3 out_view_direction;

void main(){

    float far=1000;
    float near=.001;
    vec3 position = vec4(in_position, 1.0) * in_instance_transform;

    mat4 projection=mat4(
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, far/(near-far), (near*far)/(near-far),
    0, 0, -1, 0
    );
    vec3 camera_position=vec3(0.0, 0.0, 25);
    vec3 view_direction=normalize(camera_position-vec3(0, 0, 0));
    vec3 right=normalize(cross(vec3(0, 1, 0), view_direction));
    vec3 up=cross(view_direction, right);

    mat4 view=mat4(
    vec4(right, -dot(right, camera_position)),
    vec4(up, -dot(up, camera_position)),
    vec4(view_direction, -dot(view_direction, camera_position)),
    vec4(0, 0, 0, 1));

    out_normal=in_normal*mat3(in_instance_transform);
    out_view_direction=camera_position-position;

    gl_Position=vec4(position, 1)*view*projection;
}