# Anomaly_Detection


# The goal of this project is to to detect illegitimate connections in a computer network using an autoencoder

Data set description
Data set for this project is from The Third International Knowledge Discovery and Data Mining Tools Competition at KDD-99, The Fifth International Conference on Knowledge Discovery and Data Mining. File kddCupTrain.csv with the data necessary for this project contains only one of multiple types of attacks (see below).

The competition task was building a network intrusion detector capable of distinguishing "bad" connections, called intrusions or attacks, from "good" normal connections. This database contains a variety of intrusions simulated in a military network environment.

The original KDD training dataset consists of approximately 4,900,000 single connection vectors each of which contains 41 features and is labeled as either normal or an attack, with exactly one specific attack type. The simulated attacks fall in one of the following four categories:

Denial of Service Attack (DoS): is an attack in which the attacker makes some computing or memory resource too busy or too full to handle legitimate requests, or denies legitimate users access to a machine.
User to Root Attack (U2R): is a class in which the attacker starts out with access to a normal user account on the system (perhaps gained by sniffing passwords, a dictionary attack, or social engineering) and is able to exploit some vulnerability to gain root access to the system.
Remote to Local Attack (R2L): occurs when an attacker who has the ability to send packets to a machine over a network but who does not have an account on that machine exploits some vulnerability to gain local access as a user of that machine.
Probing Attack: is an attempt to gather information about a network of computers for the apparent purpose of circumventing its security controls.
Attacks contained in the dataset:

Attack Category	Attack Type
DoS	back, land, neptune, 
pod, smurf, teardrop
U2R	buffer_overflow, loadmodule, 
perl, rootkit
R2L	ftp_write, guess_passwd, 
imap, multihop, rhf, 
spy, warezclient, warezmaster
Probe	portsweep, ipsweep, 
satan, nmap
KDD-99 features can be classified into three groups:
1) Basic features: this category encapsulates all the attributes that can be extracted from a TCP/IP connection. Most of these features leading to an implicit delay in detection.
2) Traffic features: this category includes features that are computed with respect to a window interval and is divided into two groups:

"same host" features: examine only the connections in the past 2 seconds that have the same destination host as the current connection, and calculate statistics related to protocol behavior, service, etc.
"same service" features: examine only the connections in the past 2 seconds that have the same service as the current connection.
These two types of "traffic" features are called time-based as opposed to the following connection-based type.

"connection-based" features: there are several types of slow probing attacks that scan the hosts (or ports) using a much larger time interval than 2 seconds, for example, one in every minute. As a result, these attacks do not produce intrusion patterns with a time window of 2 seconds. To detect such attacks the “same host” and “same service” features are recalculated but based on the connection window of 100 connections rather than a time window of 2 seconds. These features are called connection-based traffic features.
3) Content features: unlike most of the DoS and Probing attacks, the R2L and U2R attacks don’t have any frequent sequential intrusion patterns. This is because the DoS and Probing attacks involve many connections to some host(s) in a very short period of time. Unlike them, the R2L and U2R attacks are embedded in the data portions of the packets, and normally involve only a single connection. To detect these kinds of attacks, one needs some features in order to look for suspicious behavior in the data portion, e.g., number of failed login attempts. These features are called content features.
