function R = get_R(az, el, ct)
% function that generates rotation matrix given azimuth az, elevation el 
% and camera tilt ct
% get cos' and sin' of all angles
ca = cosd(az); sa = sind(az); 
cb = cosd(el); sb = sind(el); 
cc = cosd(ct); sc = sind(ct);
% rotation around Z by az
Ra = [ca, -sa, 0; sa, ca, 0; 0, 0, 1];
% rotation around X by el
Rb = [1, 0, 0; 0, cb, -sb; 0, sb, cb];
% rotation around Z by ct
Rc = [cc, -sc, 0; sc, cc, 0; 0, 0, 1];
% combine all
R = Rc * Rb * Ra;
