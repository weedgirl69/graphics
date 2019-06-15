#version 450

layout(location=0)in vec3 in_position;
layout(location=1)in vec3 in_normal;
layout(location=0)out vec3 out_color;

void main(){
    out_color=in_normal;
    
    float far=10;
    float near=1;
    
    mat4 projection=mat4(
        1.5,0,0,0,
        0,-1.5,0,0,
        0,0,far/(near-far),(near*far)/(near-far),
        0,0,-1,0
    );
    vec3 camera_position=vec3(2,2,2.5);
    vec3 view_direction=normalize(camera_position);//vec3(0,0,1);
    vec3 right=normalize(cross(vec3(0,1,0),view_direction));//vec3(1,0,0);
    vec3 up=cross(view_direction,right);
    
    mat4 view=mat4(
        vec4(right,-dot(right,camera_position)),
        vec4(up,-dot(up,camera_position)),
        vec4(view_direction,-dot(view_direction,camera_position)),
        vec4(0,0,0,1));
        
        gl_Position=vec4(in_position,1)*view*projection;
    }