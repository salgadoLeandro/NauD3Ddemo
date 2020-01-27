local run = false

testCam = function()
	local f = {0, 0, 250, 1}
    local g = {0, 0, 0, 0}
    local h = {0, 0, 0, 0}
    local j = {0, 0, 0, 0}
	
	getAttr("CAMERA", "MainCamera", "VIEW", 0, f)
    getAttr("CAMERA", "Dummy", "VIEW", 0, g)

    if (f[1] == g[1] and f[2] == g[2] and f[3] == g[3]) then
        run = true
		return true
	else
        setAttr("CAMERA", "Dummy", "VIEW", 0, f)
        run = false
		return false
	end
end

testCam2 = function()
    return run
end

ntestCam = function()
    return not run
end