import framework.config as config
import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder, calculate_accuracy, calculate_confusion_matrix, print_accuracy, \
    plot_confusion_matrix, plot_confusion_matrix_each_file
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances


def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def initialization_pretained_model_efficientv2s(image_pretrained_modelfile, model, verbose=1):

    model_image = torch.load(image_pretrained_modelfile, map_location=config.device)
    # print(model_image.keys())

    model_dict = model.state_dict()
    state_dict = {}
    for k, v in model_dict.items():
        # print(k, v.size())
        # image_part.stem.0.weight torch.Size([24, 3, 3, 3])
        # image_part.stem.1.weight torch.Size([24])
        # image_part.stem.1.bias torch.Size([24])

        if 'image_part.' in k:
            # print(k, v.size())  # image_part.features.0.0.weight torch.Size([128, 3, 4, 4])
            name_in_pretrain_model = k.split('image_part.')[1]
            # print(k, name_in_pretrain_model)
            # image_part.features.0.0.weight features.0.0.weight
            if name_in_pretrain_model in model_image.keys():
                if v.size() == model_image[name_in_pretrain_model].size():
                    # print(name_in_pretrain_model, v.size(), model_image[name_in_pretrain_model].size(), '\n')
                    if verbose:
                        print('Loading : ', k, ' <---- ', name_in_pretrain_model)
                    state_dict[k] = model_image[name_in_pretrain_model]
                else:
                    pass
                    if verbose:
                        print(' error size mismatch: ', k, name_in_pretrain_model, v.size(),
                          model_image[name_in_pretrain_model].size())
            else:
                pass
                if verbose:
                    print(' error layer not found: ', k, name_in_pretrain_model)

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    return model




def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=None, epochs=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name



def forward_asc_aec(model, generate_func, cuda, return_target):
    outputs = []

    if return_target:
        targets = []

    for data in generate_func:
        (batch_x, batch_y) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)  # torch.Size([16, 10])
            # batch_output, batch_output_event = all_output[0], all_output[1]
            batch_output = all_output
            batch_output = F.softmax(batch_output, dim=-1)
            outputs.append(batch_output.data.cpu().numpy())

        if return_target:
            targets.append(batch_y)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets

    return dict



def evaluate_asc_aec(model, generator, data_type, max_iteration, cuda,
                     crossentropy=False, criterion=None, label_type=None):
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # Forward
    dict = forward_asc_aec(model=model, generate_func=generate_func, cuda=cuda, return_target=True)

    # print(dict)

    outputs = dict['output']  # (audios_num, classes_num)
    targets = dict['target']  # (audios_num, classes_num)
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
    classes_num = outputs.shape[-1]
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')

    return accuracy




def training_aec_only_testing_validation(generator, model, cuda, models_dir, epochs, adamw,
                                         batch_size, check_step, lr_init = 1e-3,
                 log_path=None):
    create_folder(models_dir)

    # Optimizer
    if adamw:
        optimizer = optim.AdamW(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08)

    max_val_scene_acc = 0.000001
    save_best_model = 0

    sample_num = len(generator.train_y)
    validate = 1
    one_epoch = int(sample_num / batch_size)
    print('one_epoch: ', one_epoch, 'is 1 epoch')
    print('really batch size: ', batch_size)
    check_iter = int(one_epoch/check_step)
    print('check_step: ', check_step, '  validating every: ', check_iter,' iteration')

    val_acc_scene = []
    val_acc_scene_file = os.path.join(log_path, 'validation_acc.txt')

    # Train on mini batches
    for iteration, all_data in enumerate(generator.generate_train()):

        (batch_x, batch_y_cpu) = all_data

        train_bgn_time = time.time()

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y_cpu, cuda)

        # print(batch_x.size(), batch_y.size())

        model.train()

        optimizer.zero_grad()

        linear_output = model(batch_x)
        # print(linear_output.size())   # torch.Size([2, 7])

        x_scene_softmax = F.log_softmax(linear_output, dim=-1)
        loss_scene = F.nll_loss(x_scene_softmax, batch_y)

        loss_scene.backward()
        optimizer.step()

        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 0:
            train_fin_time = time.time()

            if validate:
                va_acc = evaluate_asc_aec(model=model,
                                          generator=generator,
                                          data_type='validate',
                                          max_iteration=None,
                                          cuda=cuda)
                val_acc_scene.append(va_acc)

                print('epoch: ', '%.2f' % (iteration / one_epoch), 'loss_s: %.5f' % float(loss_scene),
                      ' val_scene_acc: %.3f' % va_acc)
                # print('val_scene_acc: {:.3f}'.format(va_acc))

                if va_acc > max_val_scene_acc:
                    max_val_scene_acc = va_acc
                    max_val_scene_acc_iter = iteration
                    save_best_model = 1

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            print('epoch: {}, train time: {:.3f} s, validate time: {:.3f} s, max_val_scene_acc: {:.3f} , '
                  .format('%.2f' % (iteration / one_epoch), train_time, validate_time, max_val_scene_acc, ))

            np.savetxt(val_acc_scene_file, val_acc_scene, fmt='%.5f')

        # Save model
        if save_best_model:
            save_best_model = 0
            save_out_dict = {'state_dict': model.state_dict()}
            save_out_path = os.path.join(models_dir, 'best' + config.endswith)
            torch.save(save_out_dict, save_out_path)
            print('Best model saved to {}'.format(save_out_path))

        # # Reduce learning rate
        # check_itera_step = 500
        # if lr_decay and (iteration % check_itera_step == 0 > 0):
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.9

        # Stop learning
        if iteration > (epochs * one_epoch):
            final_test = 1
            if final_test:

                if validate:
                    va_acc = evaluate_asc_aec(model=model,
                                              generator=generator,
                                              data_type='validate',
                                              max_iteration=None,
                                              cuda=cuda)
                    val_acc_scene.append(va_acc)

                    print('iter: ', iteration, 'loss_s: %.5f' % float(loss_scene),
                          ' val_scene_acc: %.3f' % va_acc)
                    # print('val_scene_acc: {:.3f}'.format(va_acc))

                    if va_acc > max_val_scene_acc:
                        max_val_scene_acc = va_acc
                        max_val_scene_acc_iter = iteration

                save_out_dict = {'state_dict': model.state_dict()}
                save_out_path = os.path.join(models_dir,
                                             'final_{}_md_{}'.format('%.4f' % va_acc,
                                                                           iteration) + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Fianl Model saved to {}'.format(save_out_path))

                print('iter: ', iteration, ' val_acc: %.6f' % va_acc)

                np.savetxt(val_acc_scene_file, val_acc_scene, fmt='%.5f')

            print('iteration: ', iteration, 'epoch: ', '%.2f' % (iteration / one_epoch),
                  'max_val_scene_acc: ', max_val_scene_acc,
                  'max_val_scene_acc_itera: ', max_val_scene_acc_iter,
                  'max_val_scene_acc_epoch: ', '%.2f' % (max_val_scene_acc_iter / one_epoch)
                  )

            print('Training is done!!!')

            print('Fianl: ', iteration, '%.2f' % (iteration / check_iter),
                  ' val_scene_acc: %.6f' % va_acc,
                  )

            break






















