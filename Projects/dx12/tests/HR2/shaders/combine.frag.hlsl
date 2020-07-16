struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

Texture2D d3d : register(t0);
SamplerState s0 : register(s0);
Texture2D msk : register(t1);
SamplerState s1 : register(s1);
Texture2D dxr : register(t2);
SamplerState s2 : register(s2);
Texture2D rao : register(t3);
SamplerState s3 : register(s3);
Texture2D rts : register(t4);
SamplerState s4 : register(s4);

float4 main(VS_OUTPUT input) : SV_TARGET {

    float4 r1 = d3d.Sample(s0, input.texPos); //raster
    float4 mk = msk.Sample(s1, input.texPos); //mask
    float4 r2 = dxr.Sample(s2, input.texPos); //dxr
    float4 ao = rao.Sample(s3, input.texPos); //ambient occlusion
    float4 sh = rts.Sample(s4, input.texPos); //shadows
    
    float4 aot = float4(pow(ao.rgb, 1.0f / 3.0f), 1.0f);
    float4 sht = float4(pow(sh.rgb, 1.0f / 3.0f), 1.0f);
    float4 color = mk.r == 1.0f ? r2 : r1 * sht * aot;

    return color;
}