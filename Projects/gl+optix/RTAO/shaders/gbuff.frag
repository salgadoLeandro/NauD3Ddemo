#version 430

in Data {
    vec4 position;
    vec4 worldPos;
    vec4 worldNor;
} DataIn;

layout (location = 0) out vec4 rt1;
layout (location = 1) out vec4 rt2;

void main() {
    rt1 = DataIn.worldPos;
    rt2 = DataIn.worldNor;
}