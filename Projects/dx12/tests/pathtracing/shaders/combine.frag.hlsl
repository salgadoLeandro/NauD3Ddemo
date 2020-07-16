struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

RWTexture2D<float4> tex1 : register(u1);
Texture2D tex2 : register(t0);
SamplerState samp : register(s0);

cbuffer cCBvalues : register(b0) {
    uint frameCount;
};

float4 main (VS_OUTPUT input) : SV_TARGET {

    int2 texsize;
    tex1.GetDimensions(texsize.x, texsize.y);
    int2 imageCoords = int2(input.texPos * texsize);
    float4 c1 = tex1[imageCoords];
    float3 c2 = tex2.Sample(samp, input.texPos).rgb;
    c2 *= 1;
    
    float4 c3 = float4(c2, 1);
    float4 cFinal = (c1 * (frameCount - 1) + c3) / frameCount;

    //tex1[imageCoords] = cFinal;

    return cFinal;
    
}