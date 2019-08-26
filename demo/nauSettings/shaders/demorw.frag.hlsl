RWTexture2D<float4> rw1 : register(u1);

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float4 texCoord : TEXCOORD0;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    uint width, height;
    rw1.GetDimensions(width, height);
    uint2 loc = uint2(input.texCoord.x * width, input.texCoord.y * height);
    rw1[loc] = input.texCoord;

    return input.texCoord;
}