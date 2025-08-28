attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute vec2 aTextureCoord;
attribute mat3 aPrecomputeLT;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;
varying highp mat3 vPrecomputeLT;

void main(void) {
  // 计算裁剪空间坐标
  gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aVertexPosition, 1.0);
  
  // 计算世界空间位置并传递
  vFragPos = (uModelMatrix * vec4(aVertexPosition, 1.0)).xyz;
  // 计算世界空间法向量并传递
  vNormal = (uModelMatrix * vec4(aNormalPosition, 0.0)).xyz;
  // 传递纹理坐标
  vTextureCoord = aTextureCoord;
  // 传递PRT预计算矩阵
  vPrecomputeLT = aPrecomputeLT;
}
