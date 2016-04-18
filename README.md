# Redes Neurais

## Trabalho 1 - MLP

Trabalho realizado utilizando ambiente LINUX, de preferencia utilize o mesmo ambiente para executar o programa.

## Requisitos:

    Python 2.7
        pip

### Instalação:
Antes de rodar o programa é necessario instalar seus requisitos, como em qualquer projeto python para isso é utilizado um ambiente virtual para que as configurações originais do sistema não seja modificadas, para isso utilizaremos o virtualenv:

    sudo pip install virtualenv
        
Apos a instalação do mesmo, estraia o programa ou clone o repositorio:

    https://github.com/senechal/ssc0570-Redes-Neurais.git

rode:

    virtualenv env
                    
Isso ira criar um ambiente virtual para instalarmos os requerimentos do programa.
Instalando requerimentos:

    pip install -r requerimentos.txt
                            
### Execução
Para executar o prugrama rode:

    python run.py mlp --config=mlp_config.json --test=mlp_test.json --train=mlp_train.json
                                    
O programa utiliza esse arquivos json para configurar e apontar a localização das bases de dados.

