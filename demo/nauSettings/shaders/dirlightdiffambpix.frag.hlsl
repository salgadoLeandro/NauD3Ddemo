SamplerState s1 : register(s0);
Texture2D t1 : register(t0);

struct VS_OUTPUT {
    float4 vertexPos : SV_POSITION;
    float3 Normal : NORMAL;
    float3 LightDirection : COLOR;
    float2 TexCoord : TEXCOORD;
};

cbuffer Constantsd : register(b1) {
    float4 lightDirection;
    float4 lightColor;
    float4 diffuse;
    float4 ambient;
    float4 emission;
    float shininess;
    int texCount;
};

float4 main ( VS_OUTPUT input ) : SV_TARGET {
    float4 color;
    float4 amb;
    float4 lightIntensityDiffuse;
    float3 lightDir;
    float3 n;
    float intensity;

    if (texCount != 0 && t1.Sample(s1, input.TexCoord).a <= 0.25)
        discard;
    
    lightDir = -normalize(input.LightDirection);
    n = normalize(input.Normal);
    intensity = max(dot(lightDir, n), 0.0);

    lightIntensityDiffuse = lightColor * intensity;
    float alpha;
    if (texCount == 0) {
        color = diffuse * lightIntensityDiffuse + diffuse * 0.3 + emission;
        alpha = diffuse.a;
    }
    else {
        color = (diffuse * lightIntensityDiffuse + emission + 0.3) * t1.Sample(s1, input.TexCoord);
        alpha = t1.Sample(s1, input.TexCoord).a * diffuse.a;
    }
    return float4(float3(color.xyz), alpha);
}