#version 300 es

#ifdef GL_ES
precision mediump float;
#endif

uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform sampler2D uSampler;

uniform sampler2D uDepthMipmap;
uniform int uLastMipLevel;
uniform vec3 uLastMipSize;
uniform int uCurLevel;

in vec2 vTextureCoord;
out vec4 FragColor;

vec4 pack (float depth) {
    // 使用rgba 4字节共32位来存储z值,1个字节精度为1/256
    const vec4 bitShift = vec4(1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0);
    const vec4 bitMask = vec4(1.0/256.0, 1.0/256.0, 1.0/256.0, 0.0);
    // gl_FragCoord:片元的坐标,fract():返回数值的小数部分
    vec4 rgbaDepth = fract(depth * bitShift); //计算每个点的z值
    rgbaDepth -= rgbaDepth.gbaa * bitMask; // Cut off the value which do not fit in 8 bits
    return rgbaDepth;
}

void main(){
  if (uCurLevel == 0) {
    vec3 color = texture(uSampler, vTextureCoord).rgb;
    FragColor =vec4(color, 1.0);
    return;
  }

  ivec2 thisLevelTexelCoord = ivec2(gl_FragCoord);
  ivec2 previousLevelBaseTexelCoord = thisLevelTexelCoord * 2;

  vec4 depthTexelValues;
  depthTexelValues.x = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord, 0).r;
  depthTexelValues.y = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(1, 0), 0).r;
  depthTexelValues.z = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(1, 1), 0).r;
  depthTexelValues.w = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(0, 1), 0).r;

  float minDepth = min(min(depthTexelValues.x, depthTexelValues.y), min(depthTexelValues.z, depthTexelValues.w));

  // 如果前一级的宽度或高度为奇数，则合并额外的textels
  ivec2 u_previousLevelDimensions = ivec2(uLastMipSize.x, uLastMipSize.y);
  bool shouldIncludeExtraColumnFromPreviousLevel = (u_previousLevelDimensions.x & 1) != 0;
  bool shouldIncludeExtraRowFromPreviousLevel    = (u_previousLevelDimensions.y & 1) != 0;
  if (shouldIncludeExtraColumnFromPreviousLevel) {
    vec2 extraColumnTexelValues;
    extraColumnTexelValues.x = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(2, 0), 0).r;
    extraColumnTexelValues.y = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(2, 1), 0).r;

    // 如果宽度和高度都为奇数，那么需要包含对角值
    if (shouldIncludeExtraRowFromPreviousLevel) {
      float cornerTexelValue = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(2, 2), 0).r;
      minDepth = min(minDepth, cornerTexelValue);
    }
    minDepth = min(minDepth, min(extraColumnTexelValues.x, extraColumnTexelValues.y));
  }

  if (shouldIncludeExtraRowFromPreviousLevel) {
    vec2 extraRowTexelValues;
    extraRowTexelValues.x = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(0, 2), 0).r;
    extraRowTexelValues.y = texelFetch(uDepthMipmap, previousLevelBaseTexelCoord + ivec2(1, 2), 0).r;
    minDepth = min(minDepth, min(extraRowTexelValues.x, extraRowTexelValues.y));
  }

  FragColor = vec4(vec3(minDepth), 1.0);
}