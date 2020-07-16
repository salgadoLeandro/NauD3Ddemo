struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

Texture2D input : register(t0);
SamplerState samp : register(s0);

float4 main(VS_OUTPUT vsinput) : SV_TARGET {
    
    float4 c2 = input.Sample(samp, vsinput.texPos);
    //float lumi = (0.2126 * c2.r) + (0.0722 * c2.g) + (0.7152 * c2.b);
    float lumi = dot(c2.rgb, float3(0.2126f, 0.7152f, 0.0722f));
    return float4(lumi * 16, lumi * 16, 0.0f, 1.0f);
}