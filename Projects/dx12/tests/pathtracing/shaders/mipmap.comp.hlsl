RWTexture2D<float4> imageUnit : register(u0);
RWTexture2D<float4> mip : register(u1);

[numthreads(1, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID) {
    
    uint2 dstPos = uint2(dtID.xy);
    uint2 srcPos = dstPos * 2;
    float2 srcColor0 = imageUnit.Load(srcPos + uint2(0, 0)).xy;
    float2 srcColor1 = imageUnit.Load(srcPos + uint2(1, 0)).xy;
    float2 srcColor2 = imageUnit.Load(srcPos + uint2(0, 1)).xy;
    float2 srcColor3 = imageUnit.Load(srcPos + uint2(1, 1)).xy;

    float maxLum = max(max(srcColor0.y, srcColor1.y), max(srcColor2.y, srcColor3.y));
    float avgLum = 0.25 * (srcColor0.x + srcColor1.x + srcColor2.x + srcColor3.x);

    mip[dstPos] = float4(avgLum, maxLum, 0.0f, 0.0f);
}