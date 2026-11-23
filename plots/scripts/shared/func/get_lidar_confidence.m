function confidence = get_lidar_confidence(lid_det, timestamp, id)
% GET_LIDAR_CONFIDENCE Get LiDAR detection confidence by timestamp and ID.
%
%   timestamp: 'YYYYMMDD_HHMMSS_NANOSEC' (UTC)
%   id:        Column index in lid_det.score
%   confidence: Detection score (NaN if not found)

    % Convert timestamp to Unix seconds
    parts = sscanf(char(timestamp), '%4d%2d%2d_%2d%2d%2d_%9d');

    dt = datetime(parts(1), parts(2), parts(3), ...
        parts(4), parts(5), parts(6), 'TimeZone', 'UTC');

    unix_ts = posixtime(dt) + double(parts(7)) * 1e-9;

    % Find matching row (tolerance for floating-point precision)
    row = find(abs(lid_det.stamp(:,1) - unix_ts) < 1e-6, 1);

    % Retrieve confidence
    confidence = NaN;
    if ~isempty(row) && id+1 >= 1 && id+1 <= size(lid_det.score, 2)
        confidence = lid_det.score(row, id+1);
    end
end