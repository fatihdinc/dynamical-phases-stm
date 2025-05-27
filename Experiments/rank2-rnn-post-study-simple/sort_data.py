import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os



# Define the training function
def sort_rnn(process_idx,post,T):
    
    file_name =f'./results/post_{post}/delay_{T}/process_{process_idx}/class.txt'
    if os.path.exists(file_name):
        os.remove(file_name)

    
      # Load the image
    import fitz  # PyMuPDF

    # Specify the path to the PDF file
    if T>8:
        pdf_path = f'./results/post_{post}/delay_{T}/process_{process_idx}/epoch_1499000.pdf'
    else:
        pdf_path = f'./results/post_{post}/delay_{T}/process_{process_idx}/epoch_499000.pdf'
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    page = pdf_document[0]  # Access the first page
    pixmap = page.get_pixmap()
    
    print(f"Pixmap format: {pixmap.colorspace.n} channels")
    print(f"Image dimensions: {pixmap.width} x {pixmap.height}")
    
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape((pixmap.height, pixmap.width, pixmap.n))

    plt.figure(figsize=(12, 8))
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Optional: turn off axis
    plt.title(file_name)
    plt.show()
    
    user_input = int(input("Enter an integer (0-> No LC, 1 -> LC, 2-> no learn): "))

    
    # Write the input to the file
    with open(file_name, "w") as file:
        file.write(f"{user_input}")
        

for i in range(100):
    for post in range(2):
        for k in range(8):
            T = k*2+6
            flag = 0
            while flag == 0:
                try:
                    #sort_rnn(i,post,T) 
                    flag = 1
                except:
                    flag = 0



#%%
from scipy.stats import fisher_exact
classes = np.zeros([100,2,8])

for i in range(100):
    for post in range(2):
        for k in range(8):
            T = k*2+6
            file_name =f'./results/post_{post}/delay_{T}/process_{i}/class.txt'
            f = open(file_name, "r")
            classes[i,post,k] = f.read()


classes[classes == 2] = np.nan 
print(np.sum(np.isnan(classes),0))

perc = np.nansum(classes,0)/np.sum(~np.isnan(classes),0)

# Define significance thresholds
pval_thresholds = [0.05, 0.01, 0.001]
stars = ['*', '**', '***']

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.3
num_t = 8
x = np.arange(num_t)  # Base x positions for each t-value

# Colors for each type of post
colors = ['#1f77b4', '#ff7f0e']

# Loop through each type of post (post_0 and post_1)
for post in range(2):
    offset = (post - 0.5) * bar_width  # Offset bars for post_0 and post_1
    for t in range(num_t):
        # Statistical test
        met = classes[:, :, t]
        table = np.array([
            [np.sum(met[:, 0] == 0), np.sum(met[:, 0] == 1)],
            [np.sum(met[:, 1] == 0), np.sum(met[:, 1] == 1)],
        ])
        res = fisher_exact(table, alternative='two-sided')
        pvalue = res.pvalue * num_t  # Bonferroni correction

        # Plot bars
        ax.bar(x[t] + offset, perc[post, t], bar_width, label=f'Post {post}' if t == 0 else "", color=colors[post])

        # Add significance stars
        if pvalue < pval_thresholds[2]:
            significance = stars[2]
        elif pvalue < pval_thresholds[1]:
            significance = stars[1]
        elif pvalue < pval_thresholds[0]:
            significance = stars[0]
        else:
            significance = None

        if significance:
            ax.text(x[t] + offset, perc[post, t] + 0.05, significance,
                    ha='center', va='bottom', fontsize=10, color='black')

# Customize plot
ax.set_xlabel('Delay', fontsize=12)
ax.set_ylabel('Percentage', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'T = {t * 2 + 6}' for t in range(num_t)])
plt.tight_layout()
plt.savefig('percentages.pdf')








