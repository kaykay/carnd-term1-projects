from moviepy.editor import VideoFileClip
white_output = 'project_video_part2.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.subclip(0, 10)
white_clip.write_videofile(white_output, audio=False)
