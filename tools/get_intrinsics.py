intrinsics = ir1.profile.as_video_stream_profile().get_intrinsics()

K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])