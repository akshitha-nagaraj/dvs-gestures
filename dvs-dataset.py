import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader

# Define the path where the data will be saved
data_path = './data'
# Load the training part of the DVSgesture dataset
train_dataset = tonic.datasets.DVSGesture(save_to=data_path, train=True)
# Load the testing part of the DVSgesture dataset
test_dataset = tonic.datasets.DVSGesture(save_to=data_path, train=False)
# Access the first sample from the dataset
first_event, first_label = train_dataset[0]
# Print out the details of the first event
print("First event data:", first_event)
print("Label of the first event:", first_label)

# First event data: [(119, 113, False,       6) (121, 110,  True,      27)
#  (125, 119,  True,      34) ... ( 66, 117,  True, 8220193)
#  ( 89,  92,  True, 8220216) ( 67, 103,  True, 8220237)]
# Label of the first event: 5

#structure of events: (x, y, bool, timestamp)(x, y, bool, timestamp)
# x and y is the positions of the event on sensor grid, the boolean value indicates off or on spike, 
# (brightness change) and the timestamp is when the event occurred relative to other events

# Visualize the first event as a grid of events over time
tonic.utils.plot_event_grid(first_event)


# transformations that convert the raw event data into a format suitable for the network, 
# such as creating frames.

# Define a transformation to convert events to frames
frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=train_dataset.sensor_size, time_window=1000)
])

# Apply the transformation to the datasets
train_dataset.transform = frame_transform
test_dataset.transform = frame_transform
# The dataset can now be used in a training loop with a model designed to handle spiking data

# Cache datasets for efficient loading
cached_trainset = DiskCachedDataset(train_dataset, cache_path='./cache/dvs/train')
cached_testset = DiskCachedDataset(test_dataset, cache_path='./cache/dvs/test')

# Create data loaders
trainloader = DataLoader(cached_trainset, batch_size=128, collate_fn=tonic.collation.PadTensors())
testloader = DataLoader(cached_testset, batch_size=128, collate_fn=tonic.collation.PadTensors())

# Specify the sensor size and number of classes based on your dataset
sensor_size = (2, 128, 128)  # This should be set based on your dataset specifics
num_classes = 11  # Adjust based on actual number of classes in your dataset

model = create_snn_network(sensor_size, num_classes)
# forward_pass(model, data, device)

# Define your optimizer and loss function here
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()  # Example, adjust based on your specific needs


# for data, target in trainloader:
#         spk_rec = forward_pass(model, data, device)
        # Further processing like loss computation, backward pass, etc.

accuracy_history = []  # To store accuracy after each epoch
# Training loop
num_epochs = 10  # Set the number of epochs
for epoch in range(num_epochs):
    model.train()
    for data, targets in trainloader:
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        spk_rec = forward_pass(model, data)

        # Loss calculation
        loss = loss_fn(spk_rec, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for data, targets in testloader:
            data, targets = data.to(device), targets.to(device)
            spk_rec = model(data)
            predicted = spk_rec.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        accuracy = 100 * correct / total
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch}, Test Accuracy: {accuracy:.2f}%")

# Plot Test Accuracy
plt.ion()
plt.figure(figsize=(8, 5))  # Set figure size
plt.plot(accuracy_history, marker='o', linestyle='-', color='b')  # Line plot with markers
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show(block=True)
# plt.savefig('accuracy_plot.png')
