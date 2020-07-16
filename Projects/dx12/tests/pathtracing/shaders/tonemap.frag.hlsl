struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float2 texPos : TEXCOORD0;
};

Texture2D origTex : register(t0);
SamplerState samp0 : register(s0);
Texture2D lumiTex : register(t1);
SamplerState samp1 : register(s1);
Texture2D lumiMipTex : register(t2);
SamplerState samp2 : register(s2);

static float alpha = 0.45f;

float3 RGB2xyY(float3 rgb) {
    
    float3x3 RGB2XYZ = float3x3(0.4124, 0.3576, 0.1805,
                                0.2126, 0.7152, 0.0722,
                                0.0193, 0.1192, 0.9505);
    float3 XYZ = mul(RGB2XYZ, rgb);

    return float3(XYZ.x / (XYZ.x + XYZ.y + XYZ.z),
                  XYZ.y / (XYZ.x + XYZ.y + XYZ.z),
                  XYZ.y);
}

float3 xyY2RGB(float3 xyY) {

    float3 XYZ = float3((xyY.z / xyY.y) * xyY.x,
					     xyY.z,
					    (xyY.z / xyY.y) * (1.0 - xyY.x - xyY.y));

	float3x3 XYZ2RGB = float3x3(3.2406, -1.5372, -0.4986,
                               -0.9689,  1.8758,  0.0415, 
                                0.0557, -0.2040,  1.0570);
	
	return mul(XYZ2RGB, XYZ);
}


float4 main(VS_OUTPUT input) : SV_TARGET {

    int2 imageCoords = int2(input.texPos * float2(512, 512));
    float2 lumi = lumiTex[imageCoords].rg;
    float2 lumiAccum = lumiMipTex[int2(0, 0)].rg;
    //float2 lumiAccum = lumiMipTex.Sample(samp2, float2(0,0)).rg;

    float3 c2 = origTex.Sample(samp0, input.texPos).rgb;

    /*float l = (alpha / lumiAccum.r) * lumi.r;
    //l = (l * (1 + l / (lumiAccum.g))) / (1 + l);
    //return float4(l,l,l,1);

    float3 xyY = RGB2xyY(c2);
    xyY.z *= l;
    c2 = xyY2RGB(xyY);
    c2 = pow(c2, float3(1 / 2.2f, 1 / 2.2f, 1 / 2.2f));

    return float4(c2, 1.0f);*/
    //return float4(l,l,l,1);

    return float4(pow(c2, 1/2.2f), 1.0f);    
}

/*
float4 main(VS_OUTPUT input) : SV_TARGET {

    int2 imageCoords = int2(input.texPos * float2(512, 512));
    float2 lumi = lumiTex[imageCoords].rg;
    float2 lumiAccum = lumiMipTex[int2(0, 0)].rg;
    //float2 lumiAccum = lumiMipTex.Sample(samp2, float2(0,0)).rg;

    float3 c2 = origTex.Sample(samp0, input.texPos).rgb;

    float l = (alpha / lumiAccum.r) * lumi.r;

    l = (l * (1 + l / (lumiAccum.g))) / (1 + l);

    float3 xyY = RGB2xyY(c2);
    xyY.z *= l;
    c2 = xyY2RGB(xyY);
    c2 = pow(c2, float3(1 / 2.2f, 1 / 2.2f, 1 / 2.2f));

    return float4(c2, 1.0f);
    //return float4(l,l,l,1);
    
}
*/