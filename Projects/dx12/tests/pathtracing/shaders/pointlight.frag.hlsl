Texture2D texUnit : register(t0);
SamplerState samp : register(s0);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float3 normalV : NORMAL;
    float2 texCoordV : TEXCOORD0;
    float3 eyeV : COLOR0;
    float3 lightDirV : COLOR1;
};

struct OUTPUT_RT {
    float4 colorOut0 : SV_TARGET0;
    float4 colorOut1 : SV_TARGET1;
};

cbuffer plCBvaluesF : register(b1) {
    float4 diffuse;
    int texCount;
};

OUTPUT_RT main(VS_OUTPUT input) {

    OUTPUT_RT outp;
    
    if(texCount == 0)
        outp.colorOut0 = diffuse;
    else
        outp.colorOut0 = texUnit.Sample(samp, input.texCoordV.xy);
    outp.colorOut1 = outp.colorOut0;

    return outp;
}