struct VS_INPUT {
    float4 position : POSITION;
    float4 texCoord : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    
    VS_OUTPUT output;
    output.position = input.position;
    output.texPos = input.texCoord.xy;

    return output;
}