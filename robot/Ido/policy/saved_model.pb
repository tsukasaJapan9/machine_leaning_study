®è
Î

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8·
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:

*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:
*
dtype0
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
action_0_observationPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
j
action_0_rewardPlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
m
action_0_step_typePlaceholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ç
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_typedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_75090
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ø
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_75102
Ù
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_75124

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_75117

NoOpNoOp
á 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
value B  B 
³

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures*
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

_wrapped_policy*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
get_metadata* 
TN
VARIABLE_VALUEdense_2/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense_2/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUE
dense/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdense_1/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense_1/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*


_q_network*
* 
* 
* 
* 
* 
* 
* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
	!model*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 

"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 

'layer_with_weights-0
'layer-0
(layer_with_weights-1
(layer-1
)layer_with_weights-2
)layer-2
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
* 

!0*
* 
* 
* 
¦
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
¦
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias*
¦
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
6
Gtrace_0
Htrace_1
Itrace_2
Jtrace_3* 
6
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_3* 

0
1*

0
1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 

0
1*

0
1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 

0
1*

0
1*
* 

]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 
* 

'0
(1
)2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_75687
ü
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariabledense_2/kerneldense_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_75718Óô
û

*__inference_sequential_layer_call_fn_75356
dense_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
Ì
i
)__inference_function_with_signature_75109
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_<lambda>_74887^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
¼
5
#__inference_get_initial_state_75096

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Á
;
)__inference_function_with_signature_75097

batch_sizeü
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_get_initial_state_75096*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ÑJ
ú
'__inference_polymorphic_action_fn_75053
	time_step
time_step_1
time_step_2
time_step_3J
8qnetwork_sequential_dense_matmul_readvariableop_resource:
G
9qnetwork_sequential_dense_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_1_matmul_readvariableop_resource:

I
;qnetwork_sequential_dense_1_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_2_matmul_readvariableop_resource:
I
;qnetwork_sequential_dense_2_biasadd_readvariableop_resource:
identity¢0QNetwork/sequential/dense/BiasAdd/ReadVariableOp¢/QNetwork/sequential/dense/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_1/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_2/MatMul/ReadVariableOpt
QNetwork/sequential/dense/CastCasttime_step_3*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/QNetwork/sequential/dense/MatMul/ReadVariableOpReadVariableOp8qnetwork_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¹
 QNetwork/sequential/dense/MatMulMatMul"QNetwork/sequential/dense/Cast:y:07QNetwork/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
0QNetwork/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp9qnetwork_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
!QNetwork/sequential/dense/BiasAddBiasAdd*QNetwork/sequential/dense/MatMul:product:08QNetwork/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

QNetwork/sequential/dense/ReluRelu*QNetwork/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Ç
"QNetwork/sequential/dense_1/MatMulMatMul,QNetwork/sequential/dense/Relu:activations:09QNetwork/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ê
#QNetwork/sequential/dense_1/BiasAddBiasAdd,QNetwork/sequential/dense_1/MatMul:product:0:QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 QNetwork/sequential/dense_1/ReluRelu,QNetwork/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0É
"QNetwork/sequential/dense_2/MatMulMatMul.QNetwork/sequential/dense_1/Relu:activations:09QNetwork/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#QNetwork/sequential/dense_2/BiasAddBiasAdd,QNetwork/sequential/dense_2/MatMul:product:0:QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
Categorical/mode/ArgMaxArgMax,QNetwork/sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:¢
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¨
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp1^QNetwork/sequential/dense/BiasAdd/ReadVariableOp0^QNetwork/sequential/dense/MatMul/ReadVariableOp3^QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_1/MatMul/ReadVariableOp3^QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2d
0QNetwork/sequential/dense/BiasAdd/ReadVariableOp0QNetwork/sequential/dense/BiasAdd/ReadVariableOp2b
/QNetwork/sequential/dense/MatMul/ReadVariableOp/QNetwork/sequential/dense/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_1/MatMul/ReadVariableOp1QNetwork/sequential/dense_1/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_2/MatMul/ReadVariableOp1QNetwork/sequential/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	time_step
¾

'__inference_dense_2_layer_call_fn_75628

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75334o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì

*__inference_sequential_layer_call_fn_75528

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
B__inference_dense_1_layer_call_and_return_conditional_losses_75619

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ÇJ
ò
'__inference_polymorphic_action_fn_75184
	step_type

reward
discount
observationJ
8qnetwork_sequential_dense_matmul_readvariableop_resource:
G
9qnetwork_sequential_dense_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_1_matmul_readvariableop_resource:

I
;qnetwork_sequential_dense_1_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_2_matmul_readvariableop_resource:
I
;qnetwork_sequential_dense_2_biasadd_readvariableop_resource:
identity¢0QNetwork/sequential/dense/BiasAdd/ReadVariableOp¢/QNetwork/sequential/dense/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_1/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_2/MatMul/ReadVariableOpt
QNetwork/sequential/dense/CastCastobservation*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/QNetwork/sequential/dense/MatMul/ReadVariableOpReadVariableOp8qnetwork_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¹
 QNetwork/sequential/dense/MatMulMatMul"QNetwork/sequential/dense/Cast:y:07QNetwork/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
0QNetwork/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp9qnetwork_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
!QNetwork/sequential/dense/BiasAddBiasAdd*QNetwork/sequential/dense/MatMul:product:08QNetwork/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

QNetwork/sequential/dense/ReluRelu*QNetwork/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Ç
"QNetwork/sequential/dense_1/MatMulMatMul,QNetwork/sequential/dense/Relu:activations:09QNetwork/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ê
#QNetwork/sequential/dense_1/BiasAddBiasAdd,QNetwork/sequential/dense_1/MatMul:product:0:QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 QNetwork/sequential/dense_1/ReluRelu,QNetwork/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0É
"QNetwork/sequential/dense_2/MatMulMatMul.QNetwork/sequential/dense_1/Relu:activations:09QNetwork/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#QNetwork/sequential/dense_2/BiasAddBiasAdd,QNetwork/sequential/dense_2/MatMul:product:0:QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
Categorical/mode/ArgMaxArgMax,QNetwork/sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:¢
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¨
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp1^QNetwork/sequential/dense/BiasAdd/ReadVariableOp0^QNetwork/sequential/dense/MatMul/ReadVariableOp3^QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_1/MatMul/ReadVariableOp3^QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2d
0QNetwork/sequential/dense/BiasAdd/ReadVariableOp0QNetwork/sequential/dense/BiasAdd/ReadVariableOp2b
/QNetwork/sequential/dense/MatMul/ReadVariableOp/QNetwork/sequential/dense/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_1/MatMul/ReadVariableOp1QNetwork/sequential/dense_1/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_2/MatMul/ReadVariableOp1QNetwork/sequential/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation
¾

'__inference_dense_1_layer_call_fn_75608

inputs
unknown:


	unknown_0:

identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75318o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¼
5
#__inference_get_initial_state_75282

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
æ.

-__inference_polymorphic_distribution_fn_75279
	step_type

reward
discount
observationJ
8qnetwork_sequential_dense_matmul_readvariableop_resource:
G
9qnetwork_sequential_dense_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_1_matmul_readvariableop_resource:

I
;qnetwork_sequential_dense_1_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_2_matmul_readvariableop_resource:
I
;qnetwork_sequential_dense_2_biasadd_readvariableop_resource:
identity

identity_1

identity_2¢0QNetwork/sequential/dense/BiasAdd/ReadVariableOp¢/QNetwork/sequential/dense/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_1/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_2/MatMul/ReadVariableOpt
QNetwork/sequential/dense/CastCastobservation*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/QNetwork/sequential/dense/MatMul/ReadVariableOpReadVariableOp8qnetwork_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¹
 QNetwork/sequential/dense/MatMulMatMul"QNetwork/sequential/dense/Cast:y:07QNetwork/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
0QNetwork/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp9qnetwork_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
!QNetwork/sequential/dense/BiasAddBiasAdd*QNetwork/sequential/dense/MatMul:product:08QNetwork/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

QNetwork/sequential/dense/ReluRelu*QNetwork/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Ç
"QNetwork/sequential/dense_1/MatMulMatMul,QNetwork/sequential/dense/Relu:activations:09QNetwork/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ê
#QNetwork/sequential/dense_1/BiasAddBiasAdd,QNetwork/sequential/dense_1/MatMul:product:0:QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 QNetwork/sequential/dense_1/ReluRelu,QNetwork/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0É
"QNetwork/sequential/dense_2/MatMulMatMul.QNetwork/sequential/dense_1/Relu:activations:09QNetwork/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#QNetwork/sequential/dense_2/BiasAddBiasAdd,QNetwork/sequential/dense_2/MatMul:product:0:QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
Categorical/mode/ArgMaxArgMax,QNetwork/sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: f

Identity_1IdentityCategorical/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: ý
NoOpNoOp1^QNetwork/sequential/dense/BiasAdd/ReadVariableOp0^QNetwork/sequential/dense/MatMul/ReadVariableOp3^QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_1/MatMul/ReadVariableOp3^QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2d
0QNetwork/sequential/dense/BiasAdd/ReadVariableOp0QNetwork/sequential/dense/BiasAdd/ReadVariableOp2b
/QNetwork/sequential/dense/MatMul/ReadVariableOp/QNetwork/sequential/dense/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_1/MatMul/ReadVariableOp1QNetwork/sequential/dense_1/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_2/MatMul/ReadVariableOp1QNetwork/sequential/dense_2/MatMul/ReadVariableOp:N J
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	step_type:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namereward:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
discount:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameobservation
º
ª
#__inference_signature_wrapper_75090
discount
observation

reward
	step_type
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_function_with_signature_75068k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type


ó
B__inference_dense_1_layer_call_and_return_conditional_losses_75318

inputs0
matmul_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
º

%__inference_dense_layer_call_fn_75587

inputs
unknown:

	unknown_0:

identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
%
#__inference_signature_wrapper_75124ó
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_function_with_signature_75120*(
_construction_contextkEagerRuntime*
_input_shapes 
´
ò
E__inference_sequential_layer_call_and_return_conditional_losses_75578

inputs6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp[

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
c
#__inference_signature_wrapper_75117
unknown:	 
identity	¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_function_with_signature_75109^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
Å	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_75638

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ê

E__inference_sequential_layer_call_and_return_conditional_losses_75475
dense_input
dense_75459:

dense_75461:

dense_1_75464:


dense_1_75466:

dense_2_75469:

dense_2_75471:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallæ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_75459dense_75461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75301
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_75464dense_1_75466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75318
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75469dense_2_75471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75334w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ô
a
__inference_<lambda>_74887!
readvariableop_resource:	 
identity	¢ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: W
NoOpNoOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
»
þ
E__inference_sequential_layer_call_and_return_conditional_losses_75424

inputs
dense_75408:

dense_75410:

dense_1_75413:


dense_1_75415:

dense_2_75418:

dense_2_75420:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_75408dense_75410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75301
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_75413dense_1_75415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75318
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75418dense_2_75420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75334w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
°
)__inference_function_with_signature_75068
	step_type

reward
discount
observation
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_polymorphic_action_fn_75053k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
0/discount:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_name0/observation
Á
5
#__inference_signature_wrapper_75102

batch_size
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *2
f-R+
)__inference_function_with_signature_75097*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
[

__inference_<lambda>_74890*(
_construction_contextkEagerRuntime*
_input_shapes 
Ñ
+
)__inference_function_with_signature_75120ä
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_<lambda>_74890*(
_construction_contextkEagerRuntime*
_input_shapes 
ß
¤
__inference__traced_save_75687
file_prefix'
#savev2_variable_read_readvariableop	-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Å
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Ô
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*I
_input_shapes8
6: : :
::
:
:

:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:

_output_shapes
: 
Å	
ó
B__inference_dense_2_layer_call_and_return_conditional_losses_75334

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ì

*__inference_sequential_layer_call_fn_75511

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
þ
E__inference_sequential_layer_call_and_return_conditional_losses_75341

inputs
dense_75302:

dense_75304:

dense_1_75319:


dense_1_75321:

dense_2_75335:

dense_2_75337:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallá
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_75302dense_75304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75301
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_75319dense_1_75321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75318
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75335dense_2_75337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75334w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

*__inference_sequential_layer_call_fn_75456
dense_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:

	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_75424o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
´
ò
E__inference_sequential_layer_call_and_return_conditional_losses_75553

inputs6
$dense_matmul_readvariableop_resource:
3
%dense_biasadd_readvariableop_resource:
8
&dense_1_matmul_readvariableop_resource:

5
'dense_1_biasadd_readvariableop_resource:
8
&dense_2_matmul_readvariableop_resource:
5
'dense_2_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp[

dense/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
dense/MatMulMatMuldense/Cast:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

E__inference_sequential_layer_call_and_return_conditional_losses_75494
dense_input
dense_75478:

dense_75480:

dense_1_75483:


dense_1_75485:

dense_2_75488:

dense_2_75490:
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallæ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_75478dense_75480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75301
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_75483dense_1_75485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75318
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75488dense_2_75490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75334w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namedense_input
ð

ñ
@__inference_dense_layer_call_and_return_conditional_losses_75599

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0k
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð

ñ
@__inference_dense_layer_call_and_return_conditional_losses_75301

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0k
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
¥
!__inference__traced_restore_75718
file_prefix#
assignvariableop_variable:	 3
!assignvariableop_1_dense_2_kernel:
-
assignvariableop_2_dense_2_bias:1
assignvariableop_3_dense_kernel:
+
assignvariableop_4_dense_bias:
3
!assignvariableop_5_dense_1_kernel:

-
assignvariableop_6_dense_1_bias:


identity_8¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B Æ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ë

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: Ù
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¡K

'__inference_polymorphic_action_fn_75244
time_step_step_type
time_step_reward
time_step_discount
time_step_observationJ
8qnetwork_sequential_dense_matmul_readvariableop_resource:
G
9qnetwork_sequential_dense_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_1_matmul_readvariableop_resource:

I
;qnetwork_sequential_dense_1_biasadd_readvariableop_resource:
L
:qnetwork_sequential_dense_2_matmul_readvariableop_resource:
I
;qnetwork_sequential_dense_2_biasadd_readvariableop_resource:
identity¢0QNetwork/sequential/dense/BiasAdd/ReadVariableOp¢/QNetwork/sequential/dense/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_1/MatMul/ReadVariableOp¢2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp¢1QNetwork/sequential/dense_2/MatMul/ReadVariableOp~
QNetwork/sequential/dense/CastCasttime_step_observation*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/QNetwork/sequential/dense/MatMul/ReadVariableOpReadVariableOp8qnetwork_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¹
 QNetwork/sequential/dense/MatMulMatMul"QNetwork/sequential/dense/Cast:y:07QNetwork/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
0QNetwork/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp9qnetwork_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ä
!QNetwork/sequential/dense/BiasAddBiasAdd*QNetwork/sequential/dense/MatMul:product:08QNetwork/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

QNetwork/sequential/dense/ReluRelu*QNetwork/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:

*
dtype0Ç
"QNetwork/sequential/dense_1/MatMulMatMul,QNetwork/sequential/dense/Relu:activations:09QNetwork/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ê
#QNetwork/sequential/dense_1/BiasAddBiasAdd,QNetwork/sequential/dense_1/MatMul:product:0:QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 QNetwork/sequential/dense_1/ReluRelu,QNetwork/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
1QNetwork/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp:qnetwork_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0É
"QNetwork/sequential/dense_2/MatMulMatMul.QNetwork/sequential/dense_1/Relu:activations:09QNetwork/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp;qnetwork_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#QNetwork/sequential/dense_2/BiasAddBiasAdd,QNetwork/sequential/dense_2/MatMul:product:0:QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
Categorical/mode/ArgMaxArgMax,QNetwork/sequential/dense_2/BiasAdd:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : T
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB c
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
:\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ®
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:¢
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
Deterministic/sample/Shape_1Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
:t
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:À
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_1:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¨
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp1^QNetwork/sequential/dense/BiasAdd/ReadVariableOp0^QNetwork/sequential/dense/MatMul/ReadVariableOp3^QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_1/MatMul/ReadVariableOp3^QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2^QNetwork/sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2d
0QNetwork/sequential/dense/BiasAdd/ReadVariableOp0QNetwork/sequential/dense/BiasAdd/ReadVariableOp2b
/QNetwork/sequential/dense/MatMul/ReadVariableOp/QNetwork/sequential/dense/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_1/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_1/MatMul/ReadVariableOp1QNetwork/sequential/dense_1/MatMul/ReadVariableOp2h
2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2QNetwork/sequential/dense_2/BiasAdd/ReadVariableOp2f
1QNetwork/sequential/dense_2/MatMul/ReadVariableOp1QNetwork/sequential/dense_2/MatMul/ReadVariableOp:X T
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/
_user_specified_nametime_step/observation"µ	L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
action´
4

0/discount&
action_0_discount:0ÿÿÿÿÿÿÿÿÿ
>
0/observation-
action_0_observation:0ÿÿÿÿÿÿÿÿÿ
0
0/reward$
action_0_reward:0ÿÿÿÿÿÿÿÿÿ
6
0/step_type'
action_0_step_type:0ÿÿÿÿÿÿÿÿÿ6
action,
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:×
Í

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
get_metadata
	get_train_step


signatures"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
3
4
5"
trackable_tuple_wrapper
5
_wrapped_policy"
trackable_dict_wrapper
½
trace_0
trace_12
'__inference_polymorphic_action_fn_75184
'__inference_polymorphic_action_fn_75244±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_02ã
-__inference_polymorphic_distribution_fn_75279±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
ë
trace_02Î
#__inference_get_initial_state_75282¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
°B­
__inference_<lambda>_74890"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°B­
__inference_<lambda>_74887"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
`

action
get_initial_state
get_train_step
get_metadata"
signature_map
 :
2dense_2/kernel
:2dense_2/bias
:
2dense/kernel
:
2
dense/bias
 :

2dense_1/kernel
:
2dense_1/bias
.

_q_network"
_generic_user_object
B
'__inference_polymorphic_action_fn_75184	step_typerewarddiscountobservation"±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´B±
'__inference_polymorphic_action_fn_75244time_step/step_typetime_step/rewardtime_step/discounttime_step/observation"±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
-__inference_polymorphic_distribution_fn_75279	step_typerewarddiscountobservation"±
ª²¦
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults¢
¢ 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ßBÜ
#__inference_get_initial_state_75282
batch_size"¦
²
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
#__inference_signature_wrapper_75090
0/discount0/observation0/reward0/step_type"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
#__inference_signature_wrapper_75102
batch_size"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¿B¼
#__inference_signature_wrapper_75117"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¿B¼
#__inference_signature_wrapper_75124"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
°
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses
	!model"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
å2âß
Ö²Ò
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults¢

 
¢ 
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
å2âß
Ö²Ò
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults¢

 
¢ 
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 

'layer_with_weights-0
'layer-0
(layer_with_weights-1
(layer-1
)layer_with_weights-2
)layer-2
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_sequential
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Ý
Gtrace_0
Htrace_1
Itrace_2
Jtrace_32ò
*__inference_sequential_layer_call_fn_75356
*__inference_sequential_layer_call_fn_75511
*__inference_sequential_layer_call_fn_75528
*__inference_sequential_layer_call_fn_75456¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zGtrace_0zHtrace_1zItrace_2zJtrace_3
É
Ktrace_0
Ltrace_1
Mtrace_2
Ntrace_32Þ
E__inference_sequential_layer_call_and_return_conditional_losses_75553
E__inference_sequential_layer_call_and_return_conditional_losses_75578
E__inference_sequential_layer_call_and_return_conditional_losses_75475
E__inference_sequential_layer_call_and_return_conditional_losses_75494¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zKtrace_0zLtrace_1zMtrace_2zNtrace_3
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
é
Ttrace_02Ì
%__inference_dense_layer_call_fn_75587¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zTtrace_0

Utrace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_75599¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zUtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
ë
[trace_02Î
'__inference_dense_1_layer_call_fn_75608¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z[trace_0

\trace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_75619¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z\trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
ë
btrace_02Î
'__inference_dense_2_layer_call_fn_75628¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zbtrace_0

ctrace_02é
B__inference_dense_2_layer_call_and_return_conditional_losses_75638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zctrace_0
 "
trackable_list_wrapper
5
'0
(1
)2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bý
*__inference_sequential_layer_call_fn_75356dense_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_sequential_layer_call_fn_75511inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
*__inference_sequential_layer_call_fn_75528inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bý
*__inference_sequential_layer_call_fn_75456dense_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_75553inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_75578inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_75475dense_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_sequential_layer_call_and_return_conditional_losses_75494dense_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÙBÖ
%__inference_dense_layer_call_fn_75587inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_75599inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_dense_1_layer_call_fn_75608inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_75619inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_dense_2_layer_call_fn_75628inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_2_layer_call_and_return_conditional_losses_75638inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 9
__inference_<lambda>_74887¢

¢ 
ª " 	2
__inference_<lambda>_74890¢

¢ 
ª "ª ¢
B__inference_dense_1_layer_call_and_return_conditional_losses_75619\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_dense_1_layer_call_fn_75608O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ
¢
B__inference_dense_2_layer_call_and_return_conditional_losses_75638\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_dense_2_layer_call_fn_75628O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_75599\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 x
%__inference_dense_layer_call_fn_75587O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
P
#__inference_get_initial_state_75282)"¢
¢


batch_size 
ª "¢ é
'__inference_polymorphic_action_fn_75184½Þ¢Ú
Ò¢Î
Æ²Â
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ4
observation%"
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "R²O

PolicyStep&
action
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ 
'__inference_polymorphic_action_fn_75244å¢
ú¢ö
î²ê
TimeStep6
	step_type)&
time_step/step_typeÿÿÿÿÿÿÿÿÿ0
reward&#
time_step/rewardÿÿÿÿÿÿÿÿÿ4
discount(%
time_step/discountÿÿÿÿÿÿÿÿÿ>
observation/,
time_step/observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "R²O

PolicyStep&
action
actionÿÿÿÿÿÿÿÿÿ
state¢ 
info¢ Ì
-__inference_polymorphic_distribution_fn_75279Þ¢Ú
Ò¢Î
Æ²Â
TimeStep,
	step_type
	step_typeÿÿÿÿÿÿÿÿÿ&
reward
rewardÿÿÿÿÿÿÿÿÿ*
discount
discountÿÿÿÿÿÿÿÿÿ4
observation%"
observationÿÿÿÿÿÿÿÿÿ
¢ 
ª "®²ª

PolicyStep
actionõñ½¢¹
`
Bª?

atol 

locÿÿÿÿÿÿÿÿÿ

rtol 
JªG

allow_nan_statsp

namejDeterministic_1

validate_argsp 
¢
j
parameters
¢ 
¢
jname+tfp.distributions.Deterministic_ACTTypeSpec 
state¢ 
info¢ ¶
E__inference_sequential_layer_call_and_return_conditional_losses_75475m<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
E__inference_sequential_layer_call_and_return_conditional_losses_75494m<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
E__inference_sequential_layer_call_and_return_conditional_losses_75553h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
E__inference_sequential_layer_call_and_return_conditional_losses_75578h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_sequential_layer_call_fn_75356`<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_75456`<¢9
2¢/
%"
dense_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_75511[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_75528[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
#__inference_signature_wrapper_75090Ø¢Ô
¢ 
ÌªÈ
.

0/discount 

0/discountÿÿÿÿÿÿÿÿÿ
8
0/observation'$
0/observationÿÿÿÿÿÿÿÿÿ
*
0/reward
0/rewardÿÿÿÿÿÿÿÿÿ
0
0/step_type!
0/step_typeÿÿÿÿÿÿÿÿÿ"+ª(
&
action
actionÿÿÿÿÿÿÿÿÿ^
#__inference_signature_wrapper_7510270¢-
¢ 
&ª#
!

batch_size

batch_size "ª W
#__inference_signature_wrapper_751170¢

¢ 
ª "ª

int64
int64 	;
#__inference_signature_wrapper_75124¢

¢ 
ª "ª 