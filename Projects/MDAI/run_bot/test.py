current_frame = [(5, 10), (2, 3)]
after_curr_frame = [(20, 2), (3,5), (4,10)]
last_frame = [(-1, -5)]

vectors1 = []
vectors2 = []

for after_axes in after_curr_frame:

   for last_axes in last_frame:
      v1 = last_axes[0] - after_axes[0]
      v2 = last_axes[1] - after_axes[1]
      try:
         vectors1.append([last_axes, v1/v2])
      except:
         pass

      
for current_axes in current_frame:
   
   for after_axes in after_curr_frame:
      v1 = after_axes[0] - current_axes[0]
      v2 = after_axes[1] - current_axes[1]
      try:
         vectors2.append(v1/v2)
      except:
         pass

a = set(vectors1[1]).intersection(vectors2)

v = []
print(a)

for i in vectors1:
   for j in a:
      if j==i[1]:
         v.append([i[0], j])

print(v)