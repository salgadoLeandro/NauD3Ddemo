struct VS_INPUT {
    float4 position : POSITION;
    float4 texCoord0 : TEXCOORD;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texCoordV : TEXCOORD;
};

VS_OUTPUT main( VS_INPUT input ) {
    VS_OUTPUT output;
    output.position = input.position;
    output.texCoordV = float2(input.texCoord0.xy);
    return output;
}