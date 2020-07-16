#version 430

uniform vec4 diffuse;
uniform int texCount;
uniform sampler2D texUnit;

layout (location = 0) out vec4 outp;
layout (location = 1) out vec4 mask;
layout (location = 2) out vec4 gPos;

in Data {
    vec4 position;
    vec2 texCoord;
} DataIn;

void main() {
    if (texCount == 0)
        outp = diffuse;
    else
        outp = texture(texUnit, DataIn.texCoord);
    mask = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    gPos = vec4(DataIn.position.xyz, 0.0f);
}