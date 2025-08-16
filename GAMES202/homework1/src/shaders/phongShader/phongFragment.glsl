#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

// 遮挡物平均深度
float findBlocker(sampler2D shadowMap, vec2 uv, float zReceiver) {
  poissonDiskSamples(uv);
  float totalDepth = 0.0;
  int blockCount = 0;

  for(int i = 0;i < NUM_SAMPLES; i++){
    vec2 simpleUV = uv + poissonDisk[i] / 2048.0 * 50.0;
    float shadowMapDepth = unpack(vec4(texture2D(uShadowMap,simpleUV).rgb, 1.0));
    if(zReceiver > (shadowMapDepth)){
      totalDepth += shadowMapDepth;
      blockCount +=1;
    }
  }


  //没有遮挡
  if(blockCount ==0){
    return -1.0;
  }

  //完全遮挡
  if(blockCount==NUM_SAMPLES){
    return 2.0;
  }

	return totalDepth/float( blockCount );
}

// 计算阴影偏移值以防止阴影 acne  artifacts
// 阴影偏移是解决自遮挡问题的关键技术，通过微小偏移避免表面像素错误地遮挡自身
float Bias() {
  // 获取标准化的光源方向向量
  vec3 lightDir = normalize(uLightPos);
  // 获取标准化的表面法向量
  vec3 normal = normalize(vNormal);
  // 计算动态偏移值：
  // - 当表面与光线垂直时(dot=1)，偏移最小(0.005)
  // - 当表面与光线夹角增大时(dot减小)，偏移线性增大(最大0.05)
  float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
  return  bias;
}

float PCF(sampler2D shadowMap, vec4 coords) {
  float bias = Bias();
  float visibility = 0.0;
  float currentDepth = coords.z;
  float filterSize = 1.0 / 2048.0 * 10.0;
  // poissonDiskSamples(coords.xy);
  uniformDiskSamples(coords.xy);
  for (int i = 0; i < PCF_NUM_SAMPLES; i++) {
    vec2 textcoords = poissonDisk[i] * filterSize + coords.xy;
    float closeDepth = unpack(vec4(texture2D(shadowMap, textcoords).xyz, 1.0));
    visibility += closeDepth < currentDepth - bias ? 0.0 : 1.0;
  }
  return visibility / float(PCF_NUM_SAMPLES);
}

float PCSS(sampler2D shadowMap, vec4 coords) {
  float bias = Bias();

  // STEP 1: avgblocker depth
  float avgBlockerDepth = findBlocker(shadowMap, coords.xy, coords.z);
  if(avgBlockerDepth < 0.0){
    return 1.0;
  }

  if (avgBlockerDepth > 1.0+EPS){
    return 0.0;
  }

  // STEP 2: penumbra size
  float penumbraSize = (coords.z - avgBlockerDepth) / avgBlockerDepth;

  // STEP 3: filtering
  float sum = 0.0;
  float filterSize = 1.0 / 2048.0;
  for(int i = 0; i < NUM_SAMPLES; i++){
    vec2 simpleUV = coords.xy + poissonDisk[i] * filterSize * penumbraSize;

    float shadowMapDepth = unpack(vec4(texture2D(uShadowMap, simpleUV).rgb, 1.0)) + bias;
    sum += coords.z > shadowMapDepth ? 0.0 : 1.0;
  }

  return sum / float(NUM_SAMPLES);
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord) {
  float bias = Bias();
  vec4 depthPack = texture2D(shadowMap, shadowCoord.xy); // 从阴影贴图中采样深度值
  float depthUnpack = unpack(depthPack); // 解包深度值
  if (depthUnpack > shadowCoord.z - bias) { // 检查当前片段是否在阴影中
    return 1.0;
  }
  return 0.0;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff = uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {


  float visibility;
  // perform perspective divide 执行透视划分
  vec3 projCoords = vPositionFromLight.xyz / vPositionFromLight.w;
  // transform to [0,1] range 变换到[0,1]的范围
  vec3 shadowCoord = projCoords * 0.5 + 0.5;
  // visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}