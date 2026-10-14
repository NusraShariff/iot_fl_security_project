round = [1 2 3 4 5];
accuracy = [0.8760 0.9066 0.8638 0.8461 0.8953];
ylim([0.84 0.92]);

plot(round, accuracy, '-o', 'LineWidth', 2)
xlabel('Rounds')
ylabel('Accuracy')
title('Accuracy vs Federated Learning Rounds')
grid on