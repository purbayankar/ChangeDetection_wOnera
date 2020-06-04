# coding: utf-8

"""
Change Detection models for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

import logging
import math

import tensorflow as tf


logger = logging.getLogger(__name__)


# iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA = A.norm()
    Y = A.div(normA)
    I = tf.eye(dim, dtype=A.dtype)
    Z = tf.eye(dim, dtype=A.dtype)

    for _ in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / tf.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize, dim, _ = A.shape
    normA = A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = tf.eye(dim, dtype=A.dtype).unsqueeze(0).expand_as(A)
    Z = tf.eye(dim, dtype=A.dtype).unsqueeze(0).expand_as(A)

    for _ in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    # A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / tf.sqrt(normA)

    return A_isqrt


# deconvolve channels
class ChannelDeconv(tf.keras.Model):
    def __init__(self, block, eps=1e-2, n_iter=5, momentum=0.1, sampling_stride=3):
        super().__init__()

        self.eps = eps
        self.n_iter = n_iter
        self.momentum = momentum
        self.block = block

        self.running_mean1 = tf.zeros(block, 1)
        self.running_deconv = tf.eye(block)
        self.running_mean2 = tf.zeros(1, 1)
        self.running_var = tf.ones(1, 1)
        self.num_batches_tracked = tf.Tensor(0, dtype=tf.uint32)

        self.sampling_stride = sampling_stride

    def call(self, x, training=False) -> tf.Tensor:
        x_shape = x.shape
        if len(x.shape) == 2:
            x = tf.reshape(x, [x.shape[0], x.shape[1], 1, 1])
        if len(x.shape) == 3:
            logger.exception(f"Error! Unsupprted tensor shape {x.shape}.")

        N, C, H, W = x.size()
        B = self.block

        # take the first c channels out for deconv
        c = int(C / B) * B
        if c == 0:
            logger.exception(f"Error! block should be set smaller. Now {self.block}")

        # step 1. remove mean
        if c != C:
            x1 = tf.transpose(x[:, :c], [1, 0, 2, 3])
            x1 = tf.reshape(x1, (B, -1))
        else:
            x1 = tf.transpose(x, [1, 0, 2, 3])
            x1 = tf.reshape(x1, (B, -1))

        if (
            self.sampling_stride > 1
            and H >= self.sampling_stride
            and W >= self.sampling_stride
        ):
            x1_s = x1[:, :: self.sampling_stride ** 2]
        else:
            x1_s = x1

        mean1 = tf.math.reduce_mean(x1_s, axis=-1, keepdims=True)

        if self.num_batches_tracked == 0:
            tf.identity(mean1.de)
            self.running_mean1 = tf.identity(mean1)
        if self.training:
            self.running_mean1 *= 1 - self.momentum
            self.running_mean1 += mean1 * self.momentum
        else:
            mean1 = self.running_mean1

        x1 = x1 - mean1

        # step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * tf.eye(B, dtype=x.dtype)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked == 0:
            self.running_deconv = tf.identity(deconv)

        if training:
            self.running_deconv *= 1 - self.momentum
            self.running_deconv += deconv * self.momentum
        else:
            deconv = self.running_deconv

        x1 = deconv @ x1

        # reshape to N,c,J,W
        x1 = tf.transpose(tf.reshape(x1, (c, N, H, W)), perm=(1, 0, 2, 3))

        # normalize the remaining channels
        if c != C:
            x_tmp = tf.reshape(x[:, c:], (N, -1))
            if (
                self.sampling_stride > 1
                and H >= self.sampling_stride
                and W >= self.sampling_stride
            ):
                x_s = x_tmp[:, :: self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2 = tf.math.reduce_mean(x_s)
            var = tf.math.reduce_variance(x_s)

            if self.num_batches_tracked == 0:
                self.running_mean2 = tf.identity(mean2)
                self.running_var = tf.identity(var)

            if training:
                self.running_mean2 *= 1 - self.momentum
                self.running_mean2 += mean2 * self.momentum
                self.running_var *= 1 - self.momentum
                self.running_var += var * self.momentum
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = tf.sqrt((x[:, c:] - mean2) / (var + self.eps))
            x1 = tf.concat([x1, x_tmp], axis=1)

        if training:
            self.num_batches_tracked += 1

        if len(x_shape) == 2:
            x1 = tf.reshape(x1, x_shape)
        return x1


class Delinear(tf.keras.Model):
    """
    An alternative implementation
    """

    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        eps=1e-5,
        n_iter=5,
        momentum=0.1,
        block=512,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tf.Variable(tf.Tensor(out_features, in_features))
        if bias:
            self.bias = tf.Variable(tf.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

        if block > in_features:
            block = in_features
        else:
            if in_features % block != 0:
                block = math.gcd(block, in_features)
                logger.info(f"block size set to: {block}")
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps

        self.running_mean = tf.zeros(self.block)
        self.running_deconv = tf.eye(self.block)

    def reset_parameters(self):
        pass
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1. / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def call(self, inputs, training=False):

        if training:

            # 1. reshape
            X = tf.reshape(inputs, (-1, self.block))

            # 2. subtract mean
            X_mean = tf.math.reduce_mean(X, axis=0)
            X -= tf.expand_dims(X_mean, axis=0)
            self.running_mean *= 1 - self.momentum
            self.running_mean += X_mean * self.momentum

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = tf.eye(X.shape[1], dtype=X.dtype)

            # // addmm(D1, S, D2, beta, alpha) -> D
            # // D = beta * D1 + alpha * mm(S, D2)
            # Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
            Cov = tf.transpose(X) * self.eps + X * tf.matmul(Id, (1.0 / X.shape[0]))

            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv *= 1 - self.momentum
            self.running_deconv += deconv * self.momentum

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = tf.reshape(self.weight, (-1, self.block)) @ deconv
        b = self.bias
        if self.bias is not None:
            b -= tf.math.reduce_sum(
                tf.transpose(
                    w @ tf.expand_dims(X_mean, axis=1), (self.weight.shape[0], -1)
                ),
                axis=1,
            )

        w = tf.reshape(w, self.weight.shape)

        return tf.nn.sigmoid(tf.matmul(inputs, w) + b)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"


class FastDeconv(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
        n_iter=5,
        momentum=0.1,
        block=64,
        sampling_stride=3,
        freeze=False,
        freeze_iter=100,
    ):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter = 0
        self.track_running_stats = True
        super(FastDeconv, self).__init__(
            in_channels,
            out_channels,
            tf.tuple(kernel_size),
            tf.tuple(stride),
            tf.tuple(padding),
            tf.tuple(dilation),
            False,
            tf.tuple(0),
            groups,
            bias,
            padding_mode="zeros",
        )

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)

        if groups > 1:
            # grouped conv
            block = in_channels // groups

        self.block = block

        self.num_features = kernel_size ** 2 * block
        if groups == 1:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_deconv", torch.eye(self.num_features))
        else:
            self.register_buffer(
                "running_mean", torch.zeros(kernel_size ** 2 * in_channels)
            )
            self.register_buffer(
                "running_deconv",
                torch.eye(self.num_features).repeat(in_channels // block, 1, 1),
            )

        self.sampling_stride = sampling_stride * stride
        self.counter = 0
        self.freeze_iter = freeze_iter
        self.freeze = freeze

    def call(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen = self.freeze and (self.counter > self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter += 1
            self.counter %= self.freeze_iter * 10

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0] > 1:
                X = (
                    torch.nn.functional.unfold(
                        x,
                        self.kernel_size,
                        self.dilation,
                        self.padding,
                        self.sampling_stride,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                # channel wise
                X = (
                    x.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(-1, C)[:: self.sampling_stride ** 2, :]
                )

            if self.groups == 1:
                # (C//B*N*pixels,k*k*B)
                X = (
                    X.view(-1, self.num_features, C // B)
                    .transpose(1, 2)
                    .contiguous()
                    .view(-1, self.num_features)
                )
            else:
                X = X.view(-1, X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups == 1:
                # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1.0 / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(
                    self.num_features, dtype=X.dtype, device=X.device
                ).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(
                    self.eps, Id, 1.0 / X.shape[1], X.transpose(1, 2), X
                )

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # 4. X * deconv * conv = X * (deconv * conv)
        if self.groups == 1:
            w = (
                self.weight.view(-1, self.num_features, C // B)
                .transpose(1, 2)
                .contiguous()
                .view(-1, self.num_features)
                @ deconv
            )
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(
                self.weight.shape[0], -1
            ).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C // B, -1, self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.view(-1, self.num_features, 1))).view(
                self.bias.shape
            )

        w = w.view(self.weight.shape)
        x = tf.keras.layers.Conv2D(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x




def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    class_weights = [0.1, 0.9]
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return weighted_bce + 0.5 * (dice_coef_loss(y_true, y_pred))


def UNetPP_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = FastDeconv(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x0 = x
    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = FastDeconv(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        x = Add(name='resi' + stage)([x, x0])
    return x


def EF_UNetPP(input_shape, classes=1, deep_supervision=False):
    mode='residual'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3

    inputs = Input(shape=input_shape, name='input')
    
    conv1_1 = UNetPP_ConvUnit(inputs, stage='11', nb_filter=nb_filter[0], mode=mode)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1) 
    
    conv2_1 = UNetPP_ConvUnit(pool1, stage='21', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = UNetPP_ConvUnit(conv1_2, stage='12', nb_filter=nb_filter[0], mode=mode)

    conv3_1 = UNetPP_ConvUnit(pool2, stage='31', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = UNetPP_ConvUnit(conv2_2, stage='22', nb_filter=nb_filter[1], mode=mode)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = UNetPP_ConvUnit(conv1_3, stage='13', nb_filter=nb_filter[0], mode=mode)

    conv4_1 = UNetPP_ConvUnit(pool3, stage='41', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = UNetPP_ConvUnit(conv3_2, stage='32', nb_filter=nb_filter[2], mode=mode)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = UNetPP_ConvUnit(conv2_3, stage='23', nb_filter=nb_filter[1], mode=mode)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = UNetPP_ConvUnit(conv1_4, stage='14', nb_filter=nb_filter[0], mode=mode)

    conv5_1 = UNetPP_ConvUnit(pool4, stage='51', nb_filter=nb_filter[4], mode=mode)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = UNetPP_ConvUnit(conv4_2, stage='42', nb_filter=nb_filter[3], mode=mode)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = UNetPP_ConvUnit(conv3_3, stage='33', nb_filter=nb_filter[2], mode=mode)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = UNetPP_ConvUnit(conv2_4, stage='24', nb_filter=nb_filter[1], mode=mode)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = UNetPP_ConvUnit(conv1_5, stage='15', nb_filter=nb_filter[0], mode=mode)

    nestnet_output_1 = FastDeconv(classes, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                            padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = FastDeconv(classes, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                            padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = FastDeconv(classes, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                            padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = FastDeconv(classes, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                            padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    conv_fuse = concatenate([conv1_2, conv1_3, conv1_4, conv1_5], name='merge_fuse', axis=bn_axis)
    nestnet_output_5 = FastDeconv(classes, (1, 1), activation='sigmoid', name='output_5', kernel_initializer='he_normal',
                            padding='same', kernel_regularizer=l2(1e-4))(conv_fuse)
    
    if deep_supervision:
        model = Model(input=inputs, output=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4, nestnet_output_5])
        model.compile(optimizer=Adam(lr=1e-4),
                      loss=[weighted_bce_dice_loss, weighted_bce_dice_loss, weighted_bce_dice_loss,
                            weighted_bce_dice_loss, weighted_bce_dice_loss],
                      loss_weights=[0.5, 0.5, 0.75, 0.5, 1.0],
                      metrics=['accuracy']
                      )
    else:
        model = Model(input=inputs, output=[nestnet_output_4])
        model.compile(optimizer=Adam(lr=1e-4), loss=weighted_bce_dice_loss,
                      metrics=['accuracy'])
    return model


def UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = FastDeconv(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
#    x0 = x
#    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = FastDeconv(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage)(x)
#    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        x = Add(name='resi' + stage)([x, x0])
    return x


def EF_UNet(input_shape, classes=1, loss='bce'):
    mode = 'None'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = Input(shape=input_shape, name='input')

    conv1 = UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode)
    
    # Right side of the U-Net
    up1 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = concatenate([conv4,up1], axis=bn_axis)
    conv6 = UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode)

    up2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = concatenate([conv3,up2], axis=bn_axis)
    conv7 = UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode)

    up3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = concatenate([conv2,up3], axis=bn_axis)
    conv8 = UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode)
    
    up4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = concatenate([conv1,up4], axis=bn_axis)
    conv9 = UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode)

    # Output layer of the U-Net with a softmax activation
    output = FastDeconv(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv9)

    model = Model(input=inputs, output=output)

    if loss == 'bce':
        loss = 'binary_crossentropy'
    elif loss == 'wbce':
        loss = weighted_bce_dice_loss

    model.compile(optimizer=Adam(lr=1e-4), loss = loss, metrics = ['accuracy'])    
    
    return model
