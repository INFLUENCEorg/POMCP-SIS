import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
import os
import yaml
import pprint
import numpy as np

class RNNCoreContainer(nn.Module):
    def __init__(self, true_rnn_core):
        super().__init__()
        for para in true_rnn_core.named_parameters():
            self.register_parameter(para[0], para[1])

class Container(nn.Module):
    def __init__(self, true_rnn):
        super().__init__()
        for module in true_rnn.named_modules():
            name = module[0]
            if name == "linear_layer":
                self.add_module(name, module[1])
            elif name == "gru":
                self.gru = RNNCoreContainer(module[1])

class RNNPredictor(nn.Module):
    
  def __init__(self, input_size, output_classes, hidden_state_size, core="GRU"):
    print("core: " + core)
    super().__init__()
    self.hidden_state_size = hidden_state_size
    if core == "RNN":
        self.gru = nn.RNN(input_size=input_size, hidden_size=hidden_state_size, batch_first=True)
    elif core == "GRU":
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_state_size, batch_first=True)
    self.output_classes = output_classes
    self.output_layer_size = sum(output_classes)
    self.linear_layer = nn.Linear(hidden_state_size, self.output_layer_size)

  def forward(self, inputs):
    hidden_state = torch.zeros(1, inputs.shape[0], self.hidden_state_size)
    gru_outputs, _ = self.gru(inputs, hidden_state)
    logits = self.linear_layer(gru_outputs)
    probs = []
    count = 0
    for i, num_of_outputs in enumerate(self.output_classes):
        probs.append(torch.nn.functional.softmax(logits[:,:,count:count+num_of_outputs], dim=2))
        count += num_of_outputs
    return probs

  @torch.jit.export
  def forwardLogits(self, inputs):
    # no softmax here
    hidden_state = torch.zeros(1, inputs.shape[0], self.hidden_state_size)
    gru_outputs, _ = self.gru(inputs, hidden_state)
    logits = self.linear_layer(gru_outputs)
    the_logits = []
    count = 0
    for i, num_of_outputs in enumerate(self.output_classes):
        the_logits.append(logits[:,:,count:count+num_of_outputs])
        count += num_of_outputs
    return the_logits

  @torch.jit.export
  def recurrentForward(self, hidden_state, inputs):
    gru_outputs, hidden_state = self.gru(inputs, hidden_state)
    logits = self.linear_layer(gru_outputs)
    probs = []
    count = 0
    for i, num_of_outputs in enumerate(self.output_classes):
        probs.append(torch.nn.functional.softmax(logits[:,:,count:count+num_of_outputs], dim=2))
        count += num_of_outputs
    return probs, hidden_state

class Dataset(torch.utils.data.Dataset):
  def __init__(self, inputs, outputs, masks):
    self.inputs = torch.IntTensor(inputs).type(torch.FloatTensor)
    self.outputs = torch.IntTensor(outputs).type(torch.LongTensor)
    self.masks = torch.IntTensor(masks).type(torch.LongTensor)
    # self.inputs = inputs
    # self.outputs = outputs
    # self.masks = masks
  def __len__(self):
    return len(self.outputs)
  def __getitem__(self, idx):
    return self.inputs[idx], self.outputs[idx], self.masks[idx]
    # return torch.nn.utils.rnn.pad_sequence(self.inputs[idx], batch_first=True), torch.nn.utils.rnn.pad_sequence(self.outputs[idx], batch_first=True), torch.nn.utils.rnn.pad_sequence(self.masks[idx], batch_first=True)

# generate data for training influence predictor
def generate_data(config_path, data_folder_path):
  if "SINGULARITY_CONTAINER" not in os.environ:
    command = './run scripts/generateInfluenceLearningData.sh {} {}'.format(config_path, data_folder_path)
  else:
    command = './scripts/generateInfluenceLearningData.sh {} {}'.format(config_path, data_folder_path)
  print(command)
  print(os.system(command))

def train_influence_predictor(
    config_path,
    generate_new_data=False,
    batch_size = 128,
    lr = 0.001,
    weight_decay=5e-4,
    num_epochs = None,
    num_steps=None,
    data_path=None,
    model_save_path=None,
    model_save_interval_epoch=-1,
    core="GRU",
    save_model=True,
    model_save_interval=-1,
    split_ratio = 0.8,
    max_grad_norm=None,
    from_replay_buffer=False,
    transform=False
):

    assert(num_epochs != None or num_steps != None)

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    if model_save_path is None:
      the_path = config_path.split(".yaml")[0].replace("configs", "models")
    else:
      the_path = model_save_path

    if data_path is None:
      data_path = the_path
    
    if generate_new_data is True or os.path.exists(data_path) is False:
        generate_data(config_path, data_path)
    print("data loading path:", data_path)
    print("model saving path:", the_path)
    hidden_state_size = config["AgentComponent"][config["General"]["IDOfAgentToControl"]]["Simulator"]["InfluencePredictor"]["numberOfHiddenStates"]
    core = config["AgentComponent"][config["General"]["IDOfAgentToControl"]]["Simulator"]["InfluencePredictor"]["Type"]
    
    # read inputs and outputs from files
    if from_replay_buffer is True:
      inputs = torch.load(os.path.join(data_path,"inputs.zip")).int()
      outputs = torch.load(os.path.join(data_path,"outputs.zip")).int()
      masks = torch.load(os.path.join(data_path,"masks.zip")).int()
    else:
      inputs = torch.jit.load(os.path.join(data_path,"inputs.pt"))._parameters['0']
      outputs = torch.jit.load(os.path.join(data_path,"outputs.pt"))._parameters['0']
      # generate masks
      masks = torch.ones((inputs.shape[0], inputs.shape[1])).int()

      if transform is True:
        print("transforming")
        the_inputs = []
        the_outputs = []
        the_masks = []
        num_eps = len(inputs)
        length = len(inputs[0])
        for i_ep in range(num_eps):
          if i_ep % 10000 == 0:
            print(i_ep)
          assert(len(inputs[i_ep]) == length)
          for i_step in range(length):
            the_inputs.append(inputs[i_ep][:i_step+1])
            the_outputs.append(outputs[i_ep][:i_step+1])
            the_masks.append(torch.ones(i_step+1))
        inputs = torch.nn.utils.rnn.pad_sequence(the_inputs, batch_first=True)
        outputs = torch.nn.utils.rnn.pad_sequence(the_outputs, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(the_masks, batch_first=True, padding_value=0).int()
        print("data transformed")

    try:
      print("inputs:", inputs.shape)
      print("outputs:", outputs.shape)
      print("masks:", masks.shape)
      print("data loaded.")
    except:
      print("...")

    # split raw_inputs and output into training_inputs, training_outputs, testing_inputs, testing_outputs
    input_size = inputs[0][0].shape[0]
    output_size = outputs[0][0].shape[0]
    output_classes = []
    # now this is very slow
    for i in range(output_size):
      # _max = np.max([torch.max(outputs[z][:,i]).item() for z in range(len(outputs))])
      # _min = np.min([torch.min(outputs[z][:,i]).item() for z in range(len(outputs))])
      # _num = _max-_min+1
      _num = 2
      output_classes.append(_num)
    print("output classes:")
    pprint.pprint(output_classes)
    full_dataset_size = len(inputs)
    training_set_size = int(full_dataset_size * split_ratio)
    print("training set size: " + str(training_set_size))
    testing_set_size = full_dataset_size - training_set_size
    print("testing set size: " + str(testing_set_size)) 
    training_inputs = inputs[:training_set_size]
    training_outputs = outputs[:training_set_size]
    testing_inputs = inputs[training_set_size:]
    testing_outputs = outputs[training_set_size:]
    training_masks = masks[:training_set_size]
    testing_masks = masks[training_set_size:]
    del inputs
    del outputs
    del masks
    print("training set and testing set are split.")

    # COMPUTE THE AMOUNT OF DATA
    # perhaps just from the masks?
    # yeah we should be able to compute that from the masks
    if from_replay_buffer is True:
      amount_of_training_data = len(training_inputs)
      amount_of_test_data = len(testing_inputs)
    else:
      amount_of_training_data = training_inputs.size(0) * training_inputs.size(1)
      amount_of_test_data = testing_inputs.size(0) * testing_inputs.size(1)
    print("amount of training data:", amount_of_training_data)
    print("amount of test data:", amount_of_test_data)

    training_dataset = Dataset(training_inputs, training_outputs, training_masks)
    testing_dataset = Dataset(testing_inputs, testing_outputs, testing_masks)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    testing_dataloader = DataLoader(testing_dataset, batch_size=testing_set_size,shuffle=True, drop_last=False)
    print("dataset constructed")

    # initialize the influence predictor
    predictor = RNNPredictor(input_size, output_classes, hidden_state_size, core=core)
    print("influence predictor initialized.")

    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    def evaluate(the_predictor, dataloader):
        the_predictor.eval()
        with torch.no_grad():
            loss = 0
            for batch_inputs, batch_targets, batch_masks in dataloader:
                logits = the_predictor.forwardLogits(batch_inputs)
                for i, num_classes in enumerate(output_classes):
                    the_logits = logits[i].view(-1, num_classes)
                    the_targets = batch_targets[:,:,i].view(-1)
                    the_masks = batch_masks.view(-1)
                    non_masked_loss = loss_function(the_logits, the_targets)
                    masked_loss = torch.multiply(non_masked_loss, the_masks).mean()
                    loss += masked_loss.item()
            return loss

    epoch_losses = []
    test_losses = []
    step_losses = []

    to_terminate = False
    epoch = 0
    step = 0

    while not to_terminate:

        predictor.train()
        epoch_loss = 0
        gradient_step = 0

        if model_save_interval_epoch != -1 and epoch % model_save_interval_epoch == 0:
          # save the model
          if not os.path.exists(the_path):
            os.makedirs(the_path)
          model_path = os.path.join(the_path, "model_after_{}_epochs.pt".format(epoch))

          if core == "GRU":
            script_model = torch.jit.script(predictor)
          else:
            script_model = torch.jit.script(Container(predictor))
            print("transformed")
          torch.jit.save(script_model, open(model_path, "wb"))
          print("model saved at", model_path)

        # loop of epoch
        for batch_inputs, batch_targets, batch_masks in training_dataloader:

            if save_model and model_save_interval != -1 and step % model_save_interval == 0:
              # save the model
              if not os.path.exists(the_path):
                os.makedirs(the_path)
              model_path = os.path.join(the_path, "model_after_{}_steps.pt".format(step))

              if core == "GRU":
                script_model = torch.jit.script(predictor)
              else:
                script_model = torch.jit.script(Container(predictor))
                print("transformed")
              torch.jit.save(script_model, open(model_path, "wb"))
              print("model saved at", model_path)

            predictor.zero_grad()
            logits = predictor.forwardLogits(batch_inputs)
            loss = 0
            for i, num_classes in enumerate(output_classes):
                the_logits = logits[i].view(-1, num_classes)
                the_targets = batch_targets[:,:,i].view(-1)
                the_masks = batch_masks.view(-1)
                non_masked_loss = loss_function(the_logits, the_targets)
                masked_loss = torch.multiply(non_masked_loss, the_masks).mean()
                loss += masked_loss
            loss.backward()   
            if max_grad_norm is not None:
              torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)  
            optimizer.step()
            with torch.no_grad():
                step_losses.append(loss.item())
                epoch_loss += loss.item() * batch_inputs.size()[0] / training_set_size
            gradient_step += 1
            step += 1

            if num_steps is not None and step >= num_steps:
              to_terminate = True
              break

        test_loss = evaluate(predictor, testing_dataloader)
        test_losses.append(test_loss)
        if epoch % 100 == 0 and epoch != 0:
          print("epoch_loss {} (step {}): {}".format(epoch, step, epoch_loss))
          print("test_loss {}: {}".format(epoch, test_loss))
        epoch_losses.append(epoch_loss)  

        epoch += 1

        # decide if we want to terminate
        if num_epochs is not None and epoch >= num_epochs:
          to_terminate = True



    # plt.title("training loss")
    # plt.plot(epoch_losses)
    # plt.show()
    # plt.title("testing loss")
    # plt.plot(test_losses)
    # plt.show()
    # plt.title("step loss")
    # plt.plot(step_losses)
    # plt.show()

    if os.path.exists(the_path) is not True:
      print(the_path)
      os.makedirs(the_path)

    # save the statistics
    with open(os.path.join(the_path, "training_losses.npy"), "wb") as f:
        np.save(f, epoch_losses)
    with open(os.path.join(the_path, "testing_losses.npy"), "wb") as f:
        np.save(f, test_losses)
    with open(os.path.join(the_path, "step_losses.npy"), "wb") as f:
        np.save(f, step_losses)

    # save dataset statistics
    with open(os.path.join(the_path, "dataset_info.txt"), "w") as f:
        f.write("{} {}".format(amount_of_training_data, amount_of_test_data))

    if save_model:    
      # save the model
      model_path = os.path.join(the_path, "model.pt")

      if core == "GRU":
        script_model = torch.jit.script(predictor)
      else:
        script_model = torch.jit.script(Container(predictor))
        print("transformed")
      print(script_model)
      torch.jit.save(script_model, open(model_path, "wb"))
      print("model saved at", model_path)

    return predictor

def evaluate_influence_predictor(model_path, data_path=None, inputs=None, outputs=None, masks=None, from_replay_buffer=False):
  if inputs==None:
    if from_replay_buffer is True:
      inputs = torch.load(os.path.join(data_path,"inputs.zip")).int()
      outputs = torch.load(os.path.join(data_path,"outputs.zip")).int()
      masks = torch.load(os.path.join(data_path,"masks.zip")).int()
    else:
      inputs = torch.jit.load(os.path.join(data_path,"inputs.pt"))._parameters['0']
      outputs = torch.jit.load(os.path.join(data_path,"outputs.pt"))._parameters['0']
      # generate masks
      masks = torch.ones((inputs.shape[0], inputs.shape[1])).int()
  # print("data loaded.")
  output_size = outputs.size(-1)
  output_classes = []
  for i in range(output_size):
    _max = torch.max(outputs[:,:,i]).item()
    _min = torch.min(outputs[:,:,i]).item()
    _num = _max-_min+1
    # _num = 2
    output_classes.append(_num)

  # print("constructing dataset")
  eval_dataset = Dataset(inputs, outputs, masks)
  # print("constructing dataloader")
  eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=True, drop_last=False)
  loss_function = torch.nn.CrossEntropyLoss(reduction='none')
  def evaluate(the_predictor, dataloader):
    num_data_points = 0
    the_predictor.eval()
    with torch.no_grad():
      loss = 0.0
      for batch_inputs, batch_targets, batch_masks in dataloader:
        # print(loss)
        logits = the_predictor.forwardLogits(batch_inputs)
        for i, num_classes in enumerate(output_classes):
          the_logits = logits[i].view(-1, num_classes)
          the_targets = batch_targets[:,:,i].view(-1)
          the_masks = batch_masks.view(-1)
          non_masked_loss = loss_function(the_logits, the_targets)
          masked_loss = torch.multiply(non_masked_loss, the_masks).sum()
          num_data_points += the_masks.sum().item()
          loss += masked_loss.item()
      return loss / num_data_points
  # print("loading model")
  model = torch.jit.load(model_path)
  # print("model loaded")
  return evaluate(model, eval_dataloader)