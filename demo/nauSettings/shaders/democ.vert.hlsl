struct VS_INPUT {
    float4 position : POSITION;
    float4 texCoord : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 texCoord : TEXCOORD0;
};

VS_OUTPUT main ( VS_INPUT input ) {
    
    VS_OUTPUT output;
    output.position = input.position;
    output.texCoord = input.texCoord;
    return output;
}