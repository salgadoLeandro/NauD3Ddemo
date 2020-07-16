struct VS_INPUT {
    float4 position : POSITION;
    float4 normal: NORMAL;
    float4 texCoord : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float3 normalV : NORMAL;
    float2 texCoordV : TEXCOORD0;
    float3 eyeV : COLOR0;
    float3 lightDirV : COLOR1;
};

cbuffer plCBvaluesV : register(b0) {
    float4x4 PVM;
    float4x4 VM;
    float4 lightPos;
    float3x3 normalMatrix;
};


VS_OUTPUT main (VS_INPUT input) {
    
    VS_OUTPUT output;
    output.texCoordV = input.texCoord.xy;
    output.normalV = normalize(mul(normalMatrix, input.normal.xyz));
    
    float3 pos = float3(mul(VM, input.position).xyz);
    output.eyeV = -pos;

    float3 lposCam = float3(mul(VM, lightPos).xyz);
    output.lightDirV = lposCam - pos;
    
    output.position = mul(PVM, input.position);

    return output;
}