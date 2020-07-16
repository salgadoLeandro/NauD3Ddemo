#version 430

uniform mat4 m_pvm, m_model;
uniform vec4 cameraPos;

in vec4 position;
in vec4 normal;
in vec4 texCoord;

out Data {
    vec4 position;
    vec4 worldPos;
    vec4 worldNor;
} DataOut;

void main() {
    DataOut.position = m_pvm * position;
    DataOut.worldPos = m_model * position;
    vec3 norm = (m_model * normal).xyz;
    float w = length(DataOut.worldPos - cameraPos);
    DataOut.worldNor = vec4(norm, w);
    gl_Position = DataOut.position;
}