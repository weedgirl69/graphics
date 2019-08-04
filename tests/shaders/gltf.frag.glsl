#version 450

layout(push_constant) uniform PushConstants {
  uint material_index;
};
layout(binding = 1) uniform Materials {
  vec4 base_color_factor[1024];
};
layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_view_direction;
layout(location = 0) out vec4 out_color;

const float PI = 3.1415926535;

float microfacet_distribution(float alpha_roughness, float n_dot_h) {
  float alpha_roughness_sq = alpha_roughness * alpha_roughness;
  float f = (n_dot_h * alpha_roughness_sq - n_dot_h) * n_dot_h + 1.;
  return alpha_roughness_sq / (PI * f * f);
}

float visibility_occlusion(float alpha_roughness, float n_dot_l,
                           float n_dot_v) {
  float alpha_roughness_sq = alpha_roughness * alpha_roughness;

  float GGXV = n_dot_l * sqrt(n_dot_v * n_dot_v * (1. - alpha_roughness_sq) +
                              alpha_roughness_sq);
  float GGXL = n_dot_v * sqrt(n_dot_l * n_dot_l * (1. - alpha_roughness_sq) +
                              alpha_roughness_sq);

  float GGX = GGXV + GGXL;
  if (GGX > 0.) {
    return .5 / GGX;
  }
  return 0.;
}

vec3 specular_reflection(float v_dot_h, vec3 reflectance_0,
                         vec3 reflectance_90) {
  return reflectance_0 +
         (reflectance_90 - reflectance_0) * pow(1. - v_dot_h, 5.);
}

float srgb_to_linear(float srgb) {
  if (srgb <= .0404482362771082) {
    return srgb * 12.92;
  } else {
    return pow((srgb + .055) / 1.055, 2.4);
  }
}

vec3 srgb_to_linear(vec3 srgb) {
  return vec3(srgb_to_linear(srgb.x), srgb_to_linear(srgb.y),
              srgb_to_linear(srgb.z));
}

vec3 toneMapACES(vec3 color) {
  const float A = 2.51;
  const float B = .03;
  const float C = 2.43;
  const float D = .59;
  const float E = .14;
  return (color * (A * color + B)) / (color * (C * color + D) + E);
}

void main() {
  vec3 base_color = base_color_factor[material_index].xyz;
  float roughness = .4;
  float metallic = .9;
  vec3 f0 = vec3(.04);
  vec3 diffuse_color = base_color.rgb * (vec3(1.) - f0) * (1. - metallic);
  vec3 specular_color = mix(f0, base_color, metallic);
  float reflectance =
      max(max(specular_color.r, specular_color.g), specular_color.b);
  vec3 reflectance_0 = specular_color;
  vec3 reflectance_90 = vec3(clamp(reflectance * 50., 0., 1.));

  vec3 normal = normalize(in_normal);
  vec3 view_direction = normalize(in_view_direction);
  vec3 light_direction = normalize(vec3(-.2, 1, .5));

  float alpha_roughness = roughness * roughness;
  vec3 h = normalize(view_direction + light_direction);
  float n_dot_l = max(dot(normal, light_direction), 0);
  float n_dot_v = max(dot(normal, view_direction), 0);
  float n_dot_h = max(dot(normal, h), 0);
  float v_dot_h = max(dot(view_direction, h), 0);

  float d = microfacet_distribution(alpha_roughness, n_dot_h);
  float g = visibility_occlusion(alpha_roughness, n_dot_l, n_dot_v);
  vec3 f = specular_reflection(v_dot_h, reflectance_0, reflectance_90);
  vec3 diffuse_contribution = (1. - f) * diffuse_color / PI;
  vec3 specular_contribution = d * g * f;
  if (n_dot_l > 0. || n_dot_v > 0.) {
    vec3 light_contribution =
        4. * n_dot_l * (diffuse_contribution + specular_contribution);
    out_color = vec4(toneMapACES(light_contribution), 1.);
  } else {
    out_color = vec4(0, 0, 0, 1.);
  }
  // vec3 specular_contribution = D * Vis * F;
  // vec3 diffuseContribution = (1.0 - F) * diffuse;
}