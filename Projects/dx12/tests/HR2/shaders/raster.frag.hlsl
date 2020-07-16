struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

struct RenderTargets {
    float4 output : SV_TARGET0;
    float4 mask : SV_TARGET1;
    float4 gPos : SV_TARGET2;
};

Texture2D texUnit : register(t0);
SamplerState samp : register(s0);

cbuffer ConstantsFrag : register(b1) {
    float4 diffuse;
    int texCount;
};

RenderTargets main (VS_OUTPUT input) {

    RenderTargets rt;

    if (texCount == 0)
        rt.output = diffuse;
    else
        rt.output = texUnit.Sample(samp, input.texCoord);

    rt.mask = float4(0.0f, 0.0f, 0.0f, 0.0f);
    rt.gPos = float4(input.position.xyz, 0.0f);
    return rt;
}