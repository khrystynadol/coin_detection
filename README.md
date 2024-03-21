## Coin detection
### Project task
Given a video of a table with coins on it. The task is to count how many coins are on the table. 
Each frame of the video does not necessarily cover the entire table, but each frame contains coins. 
The bonus task is to tell the total value of the coins lying face up.
### Solution description
The task is completed using the OpenCV library for image and video processing. Here's a sequence of task execution:
1. Read the video file frame by frame, and save some of them (calculated based on the target FPS).
2. Build a panorama from the saved frames.
3. Find the edges.
4. Generate the reference circles and move them through coin image.
5. Align the reference circles to the coin edges.
7. Coins denominations classification and the amount of money calculation.
   As the coins are glinting in most videos, it is almost impossible to read their face value. Therefore, it was decided to use a KNN algorithm to rely on the size ratio of the coins, which are more visible.

Results:
