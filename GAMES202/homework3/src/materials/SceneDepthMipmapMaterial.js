class SceneDepthMipmapMaterial extends Material {
    constructor(depthTexture, vertexShader, fragmentShader) {
        super({
            'uSamper': { type: 'texture', value: depthTexture },
            'uDepthMipmap': { type: 'texture', value: null },
            'uLastMipLevel': { type: '1i', value: -1 },
            'uLastMipSize': { type: '3fv', value: null },
            'uCurLevel': { type: '1i', value: 0 },
        }, [], vertexShader, fragmentShader);
        this.notShadow = true;
    }
}

async function buildSceneDepthMipmapMaterial(depthTexture, vertexPath, fragmentPath) {
    let vertexShader = await getShaderString(vertexPath);
    let fragmentShader = await getShaderString(fragmentPath);
    return new SceneDepthMipmapMaterial(depthTexture, vertexShader, fragmentShader);
}