left = 0;
right = pi;
start_point = [13.44, 5.28];
R = 1/0.205;
center = getCircle_R([12, 16.24], 13.44, 2, 0.3, R);
turning_point = [start_point(1), center(2)]
while right - left > 0.01
mid = (left + right) / 2
 vehicle_TPBV_.x0 = turning_point(1) - (R - R * cos(mid));
 vehicle_TPBV_.y0 = turning_point(2) + R * sin(mid);
vehicle_TPBV_.theta0 = mid + pi/2;
[x, y, theta, path_length, completeness_flag] = 
SearchHybridAStarPath();
 if (completeness_flag)
 left = mid;
% break
 else
 right = mid;
 end
end
for i2 = start_point(2):0.5:turning_point(2)
 V = CreateVehiclePolygon(start_point(1), i2, pi/2);
 track(end+1, :) = [start_point(1), i2, pi/2];
 plot(V(:,1), V(:,2), 'b'); drawnow
end
for the = pi/2:0.1:vehicle_TPBV_.theta0
 xx = turning_point(1) - (R - R * cos(the - pi/2));
yy = turning_point(2) + R * sin(the - pi/2);
 V = CreateVehiclePolygon(xx, yy, the);
 track(end+1, :) = [xx, yy, the];
 plot(V(:,1), V(:,2), 'b'); drawnow
end
for ii = 1 : length(x)
 if abs(theta(ii) - 3 * pi/2) < 0.1
 break;
end
 V = CreateVehiclePolygon(x(ii), y(ii), theta(ii));
 track(end+1, :) = [x(ii), y(ii), theta(ii)];
 plot(V(:,1), V(:,2), 'b'); drawnow
end
for j = y(ii):-0.5:3.9
 V = CreateVehiclePolygon(x(ii), j, -pi/2);
 track(end+1, :) = [x(ii), j, -pi/2];
 plot(V(:,1), V(:,2), 'b'); drawnow
end
xlswrite('Q1.xlsx', track)