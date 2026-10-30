function [idx, opp_idx] = find_closest_stamp(timestamp, ego_timestamp)
%MATCH_TIMESTAMPS Efficiently match timestamps between two signals.
%
%   [idx, opp_idx] = MATCH_TIMESTAMPS(timestamp, ego_timestamp, threshold)
%   finds the closest timestamp in ego_timestamp for each element of timestamp.
%   Pairs are only kept if their absolute difference is below 'threshold'.
%
%   INPUTS:
%       timestamp     - Nx1 vector of timestamps (e.g., opponent)
%       ego_timestamp - Mx1 vector of ego timestamps
%       threshold     - scalar threshold (e.g., 1e9)
%
%   OUTPUTS:
%       idx      - indices into ego_timestamp of matched timestamps
%       opp_idx  - indices into timestamp of matched timestamps

    % Ensure column vectors
    timestamp = timestamp(:);
    ego_timestamp = ego_timestamp(:);

    % Find nearest neighbors using KD-tree (efficient)
    [idx, D] = knnsearch(ego_timestamp, timestamp);

    % Apply threshold filter
    threshold = 1e9; 
    valid = D < threshold;

    % Keep only valid matches
    idx = idx(valid);
    opp_idx = find(valid);
end
